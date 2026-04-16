#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 datasets_for_volume_3dgs 的数据集列表，批量用 Vol2Splat 渲染/导出同款数据结构。

设计目标（对齐你的历史流程）：
- 体积选择：默认扫描 datasets_root 下子目录名；若命令行传入 --datasets（逗号分隔），则只处理这些
  名称且不要求在 datasets_root 下存在同名目录（便于只跑某个 vti_cache part）
- 切块复用：如果 vti_cache_root/dataset_name 下存在 *_part_XXXX.vti，则按这些 part 逐个渲染
  （等价于复用你当年从 raw -> vti_cache 的切块/裁剪边界）
- 否则：从 volumes_root 下读取同名 raw（要求文件名可解析 dims/dtype：name_XxYxZ_dtype.raw）
- TF 渲染：每个 dataset 随机选择 band_count 个 cmap（学习 gs-datagen-project 的 CMAP_POOL）
- 输出：写到 out_root/dataset_name（或 out_root/dataset_name_part_XXXX），目录结构与 datasets_for_volume_3dgs 一致：
  - TFxx/train/*.png
  - TFxx/test/*.png
  - TFxx/transforms_train.json / transforms_test.json
  - TFxx/tf_config.json
  - TFxx/metadata.json（记录 cmap 等）
  - TFxx/point_cloud/point_cloud_normalized.ply（由 points.ply 复制而来）

注意：
- 本脚本不做“任意 ROI crop”。需要 ROI 的话请在进入 pipeline 前生成对应的 part vti（你已有 vti_cache 正是这一步产物）。
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, Optional


RE_DATASET = re.compile(
    r"^(?P<stem>.+)_(?P<X>\d+)x(?P<Y>\d+)x(?P<Z>\d+)_(?P<dtype>[A-Za-z0-9]+)(?P<suffix>_part_\d{4})?$"
)

# 学习 gs-datagen-project/scripts/batch_render_volumes.py 的配色池
CMAP_POOL = [
    "Viridis (matplotlib)",
    "Inferno (matplotlib)",
    "Plasma (matplotlib)",
    "Magma (matplotlib)",
    "Turbo",
    "Cool to Warm (Extended)",
    "Rainbow Desaturated",
    "Blue to Red Rainbow Desaturated",
]


DEFAULTS = {
    "band_count": 10,
    "num": 84,
    "test_num": 20,
    "w": 800,
    "h": 800,
    "opaque_unit": 3.0,
    "scene_bbox_size": 2.6,
    "n_points": 50000,
    "tf_mode": "linear",
    "opacity_scale": 1.0,
    "hist_eq": False,
    "gradient": False,
    "grad_opacity": 0.2,
    "tf_backend": "torch",
    "tf_device": "cuda",
    "uniform_tf_filter": False,
    "uniform_stride": 1,
    "alpha_threshold": 0.0,
    "jitter": True,
    "camera_convention": "opencv",
}


@dataclass(frozen=True)
class DatasetId:
    name: str
    stem: str
    dims_xyz: tuple[int, int, int]
    dtype: str


def _list_immediate_dirs(root: str) -> list[str]:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    names = []
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if os.path.isdir(p):
            names.append(n)
    return names


def parse_dataset_id(name: str) -> DatasetId:
    m = RE_DATASET.match(name)
    if not m:
        raise ValueError(f"无法解析数据集名：{name}（期望 name_XxYxZ_dtype 或 name_XxYxZ_dtype_part_0000）")
    stem = m.group("stem")
    X, Y, Z = int(m.group("X")), int(m.group("Y")), int(m.group("Z"))
    dtype = m.group("dtype")
    return DatasetId(name=name, stem=stem, dims_xyz=(X, Y, Z), dtype=dtype)


def _split_base_and_part(dataset_name: str) -> tuple[str, Optional[str]]:
    m = re.match(r"^(?P<base>.+?)(?P<part>_part_\d{4})$", dataset_name)
    if not m:
        return dataset_name, None
    return m.group("base"), m.group("part")


def discover_input_vtis_for_dataset(vti_cache_root: str, dataset_name: str) -> list[str]:
    """
    规则：
    - 常规：优先查 vti_cache_root/dataset_name/*.vti
    - 若 dataset_name 带 _part_XXXX：
      也会回退到 vti_cache_root/base_name/*.vti，并只取对应 part 文件
    """
    root = os.path.abspath(vti_cache_root)
    base_name, part_suffix = _split_base_and_part(dataset_name)
    candidates = [dataset_name]
    if base_name != dataset_name:
        candidates.append(base_name)

    for candidate in candidates:
        d = os.path.join(root, candidate)
        if not os.path.isdir(d):
            continue
        vtis = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".vti")]
        vtis = [os.path.abspath(p) for p in vtis if os.path.isfile(p)]
        if part_suffix is not None:
            vtis = [p for p in vtis if os.path.basename(p).endswith(f"{part_suffix}.vti")]
        if vtis:
            return vtis
    return []


def discover_input_raw_for_dataset(volumes_root: str, dataset_name: str) -> str:
    """
    在 volumes_root 下递归寻找 raw：
    - 先找 {dataset_name}.raw
    - 若 dataset_name 带 _part_XXXX，再回退找 {base_name}.raw
    """
    base_name, part_suffix = _split_base_and_part(dataset_name)
    targets = [f"{dataset_name}.raw"]
    if part_suffix is not None:
        targets.append(f"{base_name}.raw")
    volumes_root = os.path.abspath(volumes_root)
    for root, _, files in os.walk(volumes_root):
        for target in targets:
            if target in files:
                return os.path.abspath(os.path.join(root, target))
    raise FileNotFoundError(f"未找到 raw：{targets}（在 {volumes_root} 下递归搜索）")


def choose_random_cmaps(band_count: int, seed: Optional[int]) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(CMAP_POOL) for _ in range(band_count)]


def merge_sampling_export_from_yaml(cfg: dict, config_path: Optional[str]) -> dict:
    """将 YAML/JSON 中的 sampling、export 段合并进已由 build_cfg_* 生成的 cfg（后者优先保留未在 YAML 中出现的键）。"""
    if not config_path or not os.path.isfile(config_path):
        return cfg
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vol2splat.config import load_config_data

    data = load_config_data(config_path)
    if "sampling" in data and isinstance(data["sampling"], dict):
        cfg["sampling"] = {**cfg.get("sampling", {}), **data["sampling"]}
    if "export" in data and isinstance(data["export"], dict):
        cfg["export"] = {**cfg.get("export", {}), **data["export"]}
    return cfg


def load_pipeline_overrides_from_config(config_path: str) -> dict:
    """
    从 Vol2Splat YAML/JSON 配置里提取本脚本关心的参数。
    仅提取与当前 batch 脚本兼容的键，避免引入额外行为变化。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from vol2splat.config import load_config_data

    cfg = load_config_data(config_path)
    render = cfg.get("render", {}) or {}
    sampling = cfg.get("sampling", {}) or {}

    out = {}
    if "executable" in render:
        out["pvpython"] = render["executable"]
    for k in ("band_count", "num", "test_num", "w", "h", "opaque_unit", "scene_bbox_size", "tf_mode", "opacity_scale", "hist_eq", "camera_convention", "gradient", "grad_opacity"):
        if k in render:
            out[k] = render[k]
    for k in ("n_points", "tf_backend", "tf_device", "uniform_tf_filter", "uniform_stride", "alpha_threshold", "jitter"):
        if k in sampling:
            out[k] = sampling[k]
    return out


def resolve_value(args: argparse.Namespace, key: str, cfg_overrides: dict):
    # 优先级：命令行显式参数 > 配置文件 > 内置默认值
    v = getattr(args, key)
    if v is not None:
        return v
    if key in cfg_overrides:
        return cfg_overrides[key]
    return DEFAULTS[key]


def write_tf_metadata(
    tf_dir: str,
    dataset_name: str,
    dims_xyz: tuple[int, int, int],
    cmap: str,
    tf_mode: str,
    band_count: int,
    hist_eq: bool,
) -> None:
    os.makedirs(tf_dir, exist_ok=True)
    meta = {
        "volume_name": dataset_name,
        "volume_dims_xyz": list(dims_xyz),
        "tf_cmap": cmap,
        "tf_mode": tf_mode,
        "band_count": band_count,
        "hist_eq": bool(hist_eq),
    }
    with open(os.path.join(tf_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_one_case_with_vol2splat(cfg_dict: dict) -> dict:
    """
    直接调用 Vol2Splat python API（避免依赖已安装 entrypoint）。
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from vol2splat import register_builtin_plugins
    from vol2splat.config import Config
    from vol2splat.core.pipeline import run_pipeline_context

    register_builtin_plugins()
    cfg = Config.from_dict(cfg_dict)
    return run_pipeline_context(None, None, cfg)


def ensure_pointcloud_normalized(tf_dir: str) -> None:
    """
    Vol2Splat 默认写 TFxx/points.ply。为了对齐 datasets_for_volume_3dgs 的习惯，复制到：
      TFxx/point_cloud/point_cloud_normalized.ply
    """
    src = os.path.join(tf_dir, "points.ply")
    if not os.path.isfile(src):
        return
    dst_dir = os.path.join(tf_dir, "point_cloud")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "point_cloud_normalized.ply")
    shutil.copyfile(src, dst)


def iter_tf_dirs(case_out_dir: str) -> Iterable[str]:
    for n in sorted(os.listdir(case_out_dir)):
        if not n.startswith("TF"):
            continue
        p = os.path.join(case_out_dir, n)
        if os.path.isdir(p):
            yield p


def is_tf_dir_complete(tf_dir: str) -> bool:
    assert os.path.isdir(tf_dir), f"TF dir not found: {tf_dir}"
    has_tf_config = os.path.isfile(os.path.join(tf_dir, "tf_config.json"))
    has_metadata = os.path.isfile(os.path.join(tf_dir, "metadata.json"))
    has_split_json = any(
        os.path.isfile(os.path.join(tf_dir, name))
        for name in ("transforms_train.json", "transforms_test.json", "transforms_val.json")
    )
    has_train_png = os.path.isdir(os.path.join(tf_dir, "train")) and any(
        n.endswith(".png") for n in os.listdir(os.path.join(tf_dir, "train"))
    )
    has_test_png = os.path.isdir(os.path.join(tf_dir, "test")) and any(
        n.endswith(".png") for n in os.listdir(os.path.join(tf_dir, "test"))
    )
    has_images = has_train_png or has_test_png
    has_points = os.path.isfile(os.path.join(tf_dir, "points.ply")) or os.path.isfile(
        os.path.join(tf_dir, "point_cloud", "point_cloud_normalized.ply")
    )
    return has_tf_config and has_metadata and has_split_json and has_images and has_points


def write_case_done_marker(case_out_dir: str, tf_dirs: list[str]) -> None:
    payload = {
        "status": "done",
        "tf_count": len(tf_dirs),
        "tf_names": [os.path.basename(p) for p in sorted(tf_dirs)],
    }
    with open(os.path.join(case_out_dir, "_batch_done.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def is_case_output_complete(case_out_dir: str) -> bool:
    if not os.path.isdir(case_out_dir):
        return False
    marker = os.path.join(case_out_dir, "_batch_done.json")
    if os.path.isfile(marker):
        data = json.load(open(marker, "r", encoding="utf-8"))
        return int(data.get("tf_count", 0)) > 0
    tf_dirs = list(iter_tf_dirs(case_out_dir))
    if not tf_dirs:
        return False
    return all(is_tf_dir_complete(p) for p in tf_dirs)


def tf_name_to_cmap_map(cmaps: list[str]) -> dict[str, str]:
    return {f"TF{i:02d}": cmap for i, cmap in enumerate(cmaps, start=1)}


def finalize_tf_outputs(
    out_dir: str,
    run_result: dict,
    tf_cmap_map: dict[str, str],
    dataset_name: str,
    dims_xyz: tuple[int, int, int],
    tf_mode: str,
    band_count: int,
    hist_eq: bool,
) -> None:
    sampled_tasks = run_result.get("sampled_tasks") or []
    sampled_tf_names = {
        t.get("tf_name")
        for t in sampled_tasks
        if isinstance(t, dict) and t.get("tf_name")
    }
    has_named_sampling_tasks = any(
        isinstance(t, dict) and t.get("tf_name")
        for t in (run_result.get("sampling_tasks") or [])
    )

    kept_tf_dirs = []
    for tf_dir in list(iter_tf_dirs(out_dir)):
        tf_name = os.path.basename(tf_dir)
        if has_named_sampling_tasks and tf_name not in sampled_tf_names:
            print(f"[skip] remove empty TF dataset: {tf_dir}")
            shutil.rmtree(tf_dir)
            continue
        write_tf_metadata(
            tf_dir,
            dataset_name,
            dims_xyz,
            tf_cmap_map.get(tf_name),
            tf_mode,
            band_count,
            hist_eq,
        )
        ensure_pointcloud_normalized(tf_dir)
        kept_tf_dirs.append(tf_dir)
    if kept_tf_dirs:
        write_case_done_marker(out_dir, kept_tf_dirs)


def build_cfg_for_vti(
    vti_path: str,
    out_dir: str,
    cmaps: list[str],
    band_count: int,
    pvpython: str,
    num: int,
    test_num: int,
    w: int,
    h: int,
    opaque_unit: float,
    scene_bbox_size: float,
    n_points: int,
    tf_mode: str,
    opacity_scale: float,
    hist_eq: bool,
    tf_backend: str,
    tf_device: str,
    uniform_tf_filter: bool,
    uniform_stride,
    alpha_threshold: float,
    jitter: bool,
    camera_convention: str,
    gradient: bool,
    grad_opacity: float,
) -> dict:
    return {
        "io": {
            "reader": "vti",
            "path": vti_path,
        },
        "preprocess": [
            {
                "name": "canonicalize",
                "backend": "torch",
                "target_spacing": 1.0,
                "pad_to_cube": True,
                "write_vti": True,
                "vti_path": os.path.join(out_dir, "_canonical.vti"),
            }
        ],
        "render": {
            "renderer": "pv_engine",
            "path": out_dir,
            "executable": pvpython,
            "num": num,
            "w": w,
            "h": h,
            "vfov": 45,
            "test_num": test_num,
            "val_num": 0,
            "radius_scale": 0.8,
            "scene_bbox_size": scene_bbox_size,
            "cmaps": cmaps,
            "band_count": band_count,
            "shading": 0,
            "opaque_unit": opaque_unit,
            "fxaa": 1,
            "tf_mode": tf_mode,
            "opacity_scale": opacity_scale,
            "hist_eq": hist_eq,
            "gradient": gradient,
            "grad_opacity": grad_opacity,
            "camera_convention": camera_convention,
            "index": True,
            "qc": {
                "enabled": True,
                "skip_failed_tf": True,
                "min_nonzero_ratio": 0.01,
                "min_mean_intensity": 0.01,
                "min_intensity_std": 0.005,
            },
        },
        "sampling": {
            "name": "opacity",
            "n_points": n_points,
            "tf_backend": tf_backend,
            "tf_device": tf_device,
            "uniform_tf_filter": bool(uniform_tf_filter),
            "uniform_stride": uniform_stride,
            "alpha_threshold": float(alpha_threshold),
            "jitter": bool(jitter),
        },
        "export": {
            "writer": "ply",
            "path": out_dir,
        },
    }


def build_cfg_for_raw(
    raw_path: str,
    dataset_id: DatasetId,
    out_dir: str,
    cmaps: list[str],
    band_count: int,
    pvpython: str,
    num: int,
    test_num: int,
    w: int,
    h: int,
    opaque_unit: float,
    scene_bbox_size: float,
    n_points: int,
    tf_mode: str,
    opacity_scale: float,
    hist_eq: bool,
    tf_backend: str,
    tf_device: str,
    uniform_tf_filter: bool,
    uniform_stride,
    alpha_threshold: float,
    jitter: bool,
    camera_convention: str,
    gradient: bool,
    grad_opacity: float,
) -> dict:
    X, Y, Z = dataset_id.dims_xyz
    return {
        "io": {
            "reader": "raw",
            "path": raw_path,
            "shape_zyx": [Z, Y, X],
            "dtype": dataset_id.dtype,
            "order": "zyx",
            "spacing": [1.0, 1.0, 1.0],
        },
        "preprocess": [
            {
                "name": "canonicalize",
                "backend": "torch",
                "target_spacing": 1.0,
                "pad_to_cube": True,
                "write_vti": True,
                "vti_path": os.path.join(out_dir, f"{dataset_id.name}_canonical.vti"),
            }
        ],
        "render": {
            "renderer": "pv_engine",
            "path": out_dir,
            "executable": pvpython,
            "num": num,
            "w": w,
            "h": h,
            "vfov": 45,
            "test_num": test_num,
            "val_num": 0,
            "radius_scale": 0.8,
            "scene_bbox_size": scene_bbox_size,
            "cmaps": cmaps,
            "band_count": band_count,
            "shading": 0,
            "opaque_unit": opaque_unit,
            "fxaa": 1,
            "tf_mode": tf_mode,
            "opacity_scale": opacity_scale,
            "hist_eq": hist_eq,
            "gradient": gradient,
            "grad_opacity": grad_opacity,
            "camera_convention": camera_convention,
            "index": True,
            "qc": {
                "enabled": True,
                "skip_failed_tf": True,
                "min_nonzero_ratio": 0.01,
                "min_mean_intensity": 0.01,
                "min_intensity_std": 0.005,
            },
        },
        "sampling": {
            "name": "opacity",
            "n_points": n_points,
            "tf_backend": tf_backend,
            "tf_device": tf_device,
            "uniform_tf_filter": bool(uniform_tf_filter),
            "uniform_stride": uniform_stride,
            "alpha_threshold": float(alpha_threshold),
            "jitter": bool(jitter),
        },
        "export": {
            "writer": "ply",
            "path": out_dir,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="按 datasets_for_volume_3dgs 名单批量渲染（复用 vti_cache 切块）")
    ap.add_argument("--config", type=str, default=None, help="可选：Vol2Splat YAML/JSON 配置文件（如 configs/batch_canonical_3dgs.yaml）")
    ap.add_argument("--datasets", type=str, default=None, metavar="NAMES", help="逗号分隔的数据集名，仅处理这些项（不要求 datasets_root 下存在同名子目录）；省略则扫描 datasets_root")
    ap.add_argument("--datasets_root", type=str, default="/root/autodl-tmp/projects/data/datasets_for_volume_3dgs")
    ap.add_argument("--volumes_root", type=str, default="/root/autodl-tmp/projects/data/volumes")
    ap.add_argument("--vti_cache_root", type=str, default="/root/autodl-tmp/projects/data/vti_cache")
    ap.add_argument("--out_root", type=str, default="/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat")
    ap.add_argument("--pvpython", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_vti_parts", type=int, default=None, help="每个数据集最多处理前 N 个 vti part（按文件名排序）")
    ap.add_argument("--band_count", type=int, default=None)
    ap.add_argument("--num", type=int, default=None)
    ap.add_argument("--test_num", type=int, default=None)
    ap.add_argument("--w", type=int, default=None)
    ap.add_argument("--h", type=int, default=None)
    ap.add_argument("--opaque_unit", type=float, default=None)
    ap.add_argument("--scene_bbox_size", type=float, default=None)
    ap.add_argument("--n_points", type=int, default=None)
    ap.add_argument("--tf_mode", type=str, default=None)
    ap.add_argument("--opacity_scale", type=float, default=None)
    ap.add_argument("--hist_eq", action="store_true", default=None, help="启用直方图均衡 band 边界")
    ap.add_argument("--gradient", action="store_true", default=None, help="启用基于梯度的不透明度过滤")
    ap.add_argument("--grad_opacity", type=float, default=None, help="梯度阈值比例，范围 [0,1]")
    ap.add_argument("--tf_backend", type=str, default=None, choices=["auto", "torch", "numpy"])
    ap.add_argument("--tf_device", type=str, default=None)
    ap.add_argument("--uniform_tf_filter", action="store_true", default=None, help="启用均匀候选+TF(alpha)过滤采样")
    ap.add_argument("--uniform_stride", type=str, default=None, help="均匀候选步长，支持整数或 auto")
    ap.add_argument("--alpha_threshold", type=float, default=None, help="TF alpha 过滤阈值")
    ap.add_argument("--jitter", action="store_true", default=None, help="启用采样点抖动（默认按配置）")
    ap.add_argument("--no_jitter", action="store_true", default=None, help="关闭采样点抖动")
    ap.add_argument("--camera_convention", type=str, default=None, choices=["opengl", "opencv"])
    ap.add_argument("--skip_existing", action="store_true", default=True, help="已存在且完整的输出目录直接跳过（默认开启）")
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing", help="即使已存在也重新生成")
    ap.add_argument("--force", action="store_true", help="强制重跑，等价于 --no_skip_existing")
    args = ap.parse_args()

    cfg_overrides = {}
    if args.config:
        cfg_overrides = load_pipeline_overrides_from_config(args.config)

    # 解析最终参数（CLI > config > default）
    args.pvpython = args.pvpython if args.pvpython is not None else cfg_overrides.get("pvpython", "pvpython")
    args.band_count = int(resolve_value(args, "band_count", cfg_overrides))
    args.num = int(resolve_value(args, "num", cfg_overrides))
    args.test_num = int(resolve_value(args, "test_num", cfg_overrides))
    args.w = int(resolve_value(args, "w", cfg_overrides))
    args.h = int(resolve_value(args, "h", cfg_overrides))
    args.opaque_unit = float(resolve_value(args, "opaque_unit", cfg_overrides))
    args.scene_bbox_size = float(resolve_value(args, "scene_bbox_size", cfg_overrides))
    args.n_points = int(resolve_value(args, "n_points", cfg_overrides))
    args.tf_mode = str(resolve_value(args, "tf_mode", cfg_overrides))
    args.opacity_scale = float(resolve_value(args, "opacity_scale", cfg_overrides))
    args.hist_eq = bool(resolve_value(args, "hist_eq", cfg_overrides))
    args.gradient = bool(resolve_value(args, "gradient", cfg_overrides))
    args.grad_opacity = float(resolve_value(args, "grad_opacity", cfg_overrides))
    args.tf_backend = str(resolve_value(args, "tf_backend", cfg_overrides))
    args.tf_device = str(resolve_value(args, "tf_device", cfg_overrides))
    args.uniform_tf_filter = bool(resolve_value(args, "uniform_tf_filter", cfg_overrides))
    stride_val = resolve_value(args, "uniform_stride", cfg_overrides)
    args.uniform_stride = str(stride_val) if isinstance(stride_val, str) else int(stride_val)
    args.alpha_threshold = float(resolve_value(args, "alpha_threshold", cfg_overrides))
    if args.no_jitter is True:
        args.jitter = False
    else:
        args.jitter = bool(resolve_value(args, "jitter", cfg_overrides))
    args.camera_convention = str(resolve_value(args, "camera_convention", cfg_overrides))
    if args.max_vti_parts is not None:
        args.max_vti_parts = int(args.max_vti_parts)
        assert args.max_vti_parts > 0, "--max_vti_parts must be > 0"
    if args.force:
        args.skip_existing = False

    os.makedirs(args.out_root, exist_ok=True)

    if args.datasets:
        dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        assert dataset_names, "--datasets 解析后为空（请使用逗号分隔名称，如 a,b,c）"
    else:
        dataset_names = _list_immediate_dirs(args.datasets_root)
    if args.limit is not None:
        dataset_names = dataset_names[: int(args.limit)]

    print(f"[batch] datasets_root:   {args.datasets_root}")
    print(f"[batch] volumes_root:    {args.volumes_root}")
    print(f"[batch] vti_cache_root:  {args.vti_cache_root}")
    print(f"[batch] out_root:        {args.out_root}")
    print(f"[batch] skip_existing:  {args.skip_existing}")
    print(f"[batch] max_vti_parts:  {args.max_vti_parts}")
    print(f"[batch] datasets:        {len(dataset_names)}  ({', '.join(dataset_names)})")

    for idx, dataset_name in enumerate(dataset_names, start=1):
        dataset_id = parse_dataset_id(dataset_name)

        vtis = discover_input_vtis_for_dataset(args.vti_cache_root, dataset_name)
        if vtis and args.max_vti_parts is not None:
            vtis = vtis[: args.max_vti_parts]
        if vtis:
            print(f"\n=== [{idx}/{len(dataset_names)}] {dataset_name}: use vti_cache parts ({len(vtis)}) ===")
            for part_idx, vti_path in enumerate(vtis, start=1):
                part_stem = os.path.splitext(os.path.basename(vti_path))[0]  # includes _part_XXXX
                out_dir = os.path.join(args.out_root, part_stem)
                if args.skip_existing and is_case_output_complete(out_dir):
                    print(f"[skip] already complete: {part_stem}")
                    continue
                os.makedirs(out_dir, exist_ok=True)
                cmaps = choose_random_cmaps(
                    args.band_count,
                    None if args.seed is None else (int(args.seed) + idx * 100000 + part_idx),
                )

                cfg = build_cfg_for_vti(
                    vti_path=vti_path,
                    out_dir=out_dir,
                    cmaps=cmaps,
                    band_count=args.band_count,
                    pvpython=args.pvpython,
                    num=args.num,
                    test_num=args.test_num,
                    w=args.w,
                    h=args.h,
                    opaque_unit=args.opaque_unit,
                    scene_bbox_size=args.scene_bbox_size,
                    n_points=args.n_points,
                    tf_mode=args.tf_mode,
                    opacity_scale=args.opacity_scale,
                    hist_eq=args.hist_eq,
                    tf_backend=args.tf_backend,
                    tf_device=args.tf_device,
                    uniform_tf_filter=args.uniform_tf_filter,
                    uniform_stride=args.uniform_stride,
                    alpha_threshold=args.alpha_threshold,
                    jitter=args.jitter,
                    camera_convention=args.camera_convention,
                    gradient=args.gradient,
                    grad_opacity=args.grad_opacity,
                )
                cfg = merge_sampling_export_from_yaml(cfg, args.config)
                with open(os.path.join(out_dir, "_vol2splat_cfg.json"), "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)

                run_result = run_one_case_with_vol2splat(cfg)
                finalize_tf_outputs(
                    out_dir=out_dir,
                    run_result=run_result,
                    tf_cmap_map=tf_name_to_cmap_map(cmaps),
                    dataset_name=part_stem,
                    dims_xyz=dataset_id.dims_xyz,
                    tf_mode=args.tf_mode,
                    band_count=args.band_count,
                    hist_eq=args.hist_eq,
                )
        else:
            print(f"\n=== [{idx}/{len(dataset_names)}] {dataset_name}: use raw ===")
            raw_path = discover_input_raw_for_dataset(args.volumes_root, dataset_name)
            out_dir = os.path.join(args.out_root, dataset_name)
            if args.skip_existing and is_case_output_complete(out_dir):
                print(f"[skip] already complete: {dataset_name}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            cmaps = choose_random_cmaps(
                args.band_count,
                None if args.seed is None else (int(args.seed) + idx),
            )

            cfg = build_cfg_for_raw(
                raw_path=raw_path,
                dataset_id=dataset_id,
                out_dir=out_dir,
                cmaps=cmaps,
                band_count=args.band_count,
                pvpython=args.pvpython,
                num=args.num,
                test_num=args.test_num,
                w=args.w,
                h=args.h,
                opaque_unit=args.opaque_unit,
                scene_bbox_size=args.scene_bbox_size,
                n_points=args.n_points,
                tf_mode=args.tf_mode,
                opacity_scale=args.opacity_scale,
                hist_eq=args.hist_eq,
                tf_backend=args.tf_backend,
                tf_device=args.tf_device,
                uniform_tf_filter=args.uniform_tf_filter,
                uniform_stride=args.uniform_stride,
                alpha_threshold=args.alpha_threshold,
                jitter=args.jitter,
                camera_convention=args.camera_convention,
                gradient=args.gradient,
                grad_opacity=args.grad_opacity,
            )
            cfg = merge_sampling_export_from_yaml(cfg, args.config)
            with open(os.path.join(out_dir, "_vol2splat_cfg.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)

            run_result = run_one_case_with_vol2splat(cfg)
            finalize_tf_outputs(
                out_dir=out_dir,
                run_result=run_result,
                tf_cmap_map=tf_name_to_cmap_map(cmaps),
                dataset_name=dataset_name,
                dims_xyz=dataset_id.dims_xyz,
                tf_mode=args.tf_mode,
                band_count=args.band_count,
                hist_eq=args.hist_eq,
            )

    print("\n[batch] done")


if __name__ == "__main__":
    main()

