import copy
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# 与 scipts/volume_splitter.py 所用配色池对齐，用于 cmaps_random
DEFAULT_CMAP_POOL: List[str] = [
    "Viridis (matplotlib)",
    "Inferno (matplotlib)",
    "Plasma (matplotlib)",
    "Magma (matplotlib)",
    "Turbo",
    "Cool to Warm (Extended)",
    "Rainbow Desaturated",
    "Blue to Red Rainbow Desaturated",
]

RE_DATASET = re.compile(
    r"^(?P<stem>.+)_(?P<X>\d+)x(?P<Y>\d+)x(?P<Z>\d+)_(?P<dtype>[A-Za-z0-9]+)(?P<suffix>_part_\d{4})?$"
)

from .config import Config, load_config_data
from .core.pipeline import run_pipeline_context


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _input_stem(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _default_patterns_for_reader(reader: str) -> List[str]:
    reader = str(reader).lower()
    if reader in {"nii", "nii.gz", "nifti"}:
        return ["*.nii.gz", "*.nii"]
    if reader == "vti":
        return ["*.vti"]
    if reader == "raw":
        return ["*.raw"]
    return ["*"]


def _discover_input_for_case(case_dir: Path, patterns: Iterable[str], filename: str | None = None) -> str | None:
    if filename:
        path = case_dir / filename
        return str(path.resolve()) if path.exists() else None
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(case_dir.glob(pattern)))
    if not matches:
        return None
    matches = sorted({path.resolve() for path in matches}, key=lambda p: str(p))
    preferred = [path for path in matches if path.name in {"ct.nii.gz", "ct.nii", "volume.vti"}]
    return str((preferred[0] if preferred else matches[0]))


def discover_case_inputs(raw_root: str, reader: str, case_glob: str = "s*", filename: str | None = None) -> List[Dict[str, str]]:
    raw_root_path = Path(raw_root).resolve()
    patterns = _default_patterns_for_reader(reader)
    cases: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in raw_root_path.glob(case_glob) if path.is_dir()):
        input_path = _discover_input_for_case(case_dir, patterns, filename=filename)
        if not input_path:
            continue
        case_id = case_dir.name
        cases.append({
            "case_id": case_id,
            "case_dir": str(case_dir),
            "input_path": input_path,
            "input_name": os.path.basename(input_path),
            "input_stem": _input_stem(input_path),
        })
    return cases


def _list_immediate_dirs(root: str) -> List[str]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(str(root_path))
    return sorted(path.name for path in root_path.iterdir() if path.is_dir())


def _parse_dataset_id(name: str) -> Dict[str, Any]:
    m = RE_DATASET.match(name)
    if not m:
        raise ValueError(f"Cannot parse dataset name: {name} (expected name_XxYxZ_dtype or name_XxYxZ_dtype_part_0000)")
    return {
        "name": name,
        "stem": m.group("stem"),
        "dims_xyz": (int(m.group("X")), int(m.group("Y")), int(m.group("Z"))),
        "dtype": m.group("dtype"),
    }


def _split_base_and_part(dataset_name: str) -> tuple[str, Optional[str]]:
    m = re.match(r"^(?P<base>.+?)(?P<part>_part_\d{4})$", dataset_name)
    if not m:
        return dataset_name, None
    return m.group("base"), m.group("part")


def _discover_input_vtis_for_dataset(vti_cache_root: str, dataset_name: str) -> List[str]:
    root = Path(vti_cache_root).resolve()
    base_name, part_suffix = _split_base_and_part(dataset_name)
    candidates = [dataset_name]
    if base_name != dataset_name:
        candidates.append(base_name)

    for candidate in candidates:
        d = root / candidate
        if not d.is_dir():
            continue
        vtis = sorted(path.resolve() for path in d.glob("*.vti") if path.is_file())
        if part_suffix is not None:
            vtis = [path for path in vtis if path.name.endswith(f"{part_suffix}.vti")]
        if vtis:
            return [str(path) for path in vtis]
    return []


def _discover_input_raw_for_dataset(volumes_root: str, dataset_name: str) -> str:
    base_name, part_suffix = _split_base_and_part(dataset_name)
    targets = [f"{dataset_name}.raw"]
    if part_suffix is not None:
        targets.append(f"{base_name}.raw")
    root = Path(volumes_root).resolve()
    for target in targets:
        matched = sorted(path.resolve() for path in root.rglob(target) if path.is_file())
        if matched:
            return str(matched[0])
    raise FileNotFoundError(f"Raw file not found for dataset {dataset_name}; expected one of {targets} under {root}")


def discover_dataset_inputs(
    datasets: str | None,
    datasets_root: str,
    volumes_root: str,
    vti_cache_root: str,
    max_vti_parts: int | None = None,
    vti_only: bool = False,
) -> List[Dict[str, Any]]:
    if datasets:
        dataset_names = [s.strip() for s in datasets.split(",") if s.strip()]
    else:
        dataset_names = _list_immediate_dirs(datasets_root)
    if not dataset_names:
        return []
    if max_vti_parts is not None and max_vti_parts <= 0:
        raise ValueError(f"max_vti_parts must be > 0, got {max_vti_parts}")

    cases: List[Dict[str, Any]] = []
    for dataset_name in dataset_names:
        dataset_id = _parse_dataset_id(dataset_name)
        vtis = _discover_input_vtis_for_dataset(vti_cache_root, dataset_name)
        if max_vti_parts is not None:
            vtis = vtis[:max_vti_parts]

        if vtis:
            for vti_path in vtis:
                part_stem = Path(vti_path).stem
                cases.append({
                    "case_id": part_stem,
                    "output_name": part_stem,
                    "dataset_name": dataset_name,
                    "dataset_dims_xyz": list(dataset_id["dims_xyz"]),
                    "dataset_dtype": dataset_id["dtype"],
                    "input_path": vti_path,
                    "input_name": os.path.basename(vti_path),
                    "input_stem": _input_stem(vti_path),
                    "input_reader": "vti",
                })
            continue
        if vti_only:
            continue

        raw_path = _discover_input_raw_for_dataset(volumes_root, dataset_name)
        x, y, z = dataset_id["dims_xyz"]
        cases.append({
            "case_id": dataset_name,
            "output_name": dataset_name,
            "dataset_name": dataset_name,
            "dataset_dims_xyz": [x, y, z],
            "dataset_dtype": dataset_id["dtype"],
            "input_path": raw_path,
            "input_name": os.path.basename(raw_path),
            "input_stem": _input_stem(raw_path),
            "input_reader": "raw",
            "io_override": {
                "shape_zyx": [z, y, x],
                "dtype": dataset_id["dtype"],
                "order": "zyx",
                "spacing": [1.0, 1.0, 1.0],
            },
        })
    return cases


def discover_dataset_inputs_from_source_root(
    source_root: str,
    datasets: str | None = None,
    max_vti_parts: int | None = None,
) -> List[Dict[str, Any]]:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    if max_vti_parts is not None and max_vti_parts <= 0:
        raise ValueError(f"max_vti_parts must be > 0, got {max_vti_parts}")

    selected_names = {s.strip() for s in datasets.split(",") if s.strip()} if datasets else None
    candidate_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    vti_dirs = [d for d in candidate_dirs if any(d.glob("*.vti"))]

    cases: List[Dict[str, Any]] = []
    if vti_dirs:
        for dataset_dir in vti_dirs:
            dataset_name = dataset_dir.name
            if selected_names is not None and dataset_name not in selected_names:
                continue
            dataset_dims = None
            dataset_dtype = None
            try:
                parsed = _parse_dataset_id(dataset_name)
                dataset_dims = list(parsed["dims_xyz"])
                dataset_dtype = parsed["dtype"]
            except Exception:
                pass
            vtis = sorted(path.resolve() for path in dataset_dir.glob("*.vti") if path.is_file())
            if max_vti_parts is not None:
                vtis = vtis[:max_vti_parts]
            for vti_path in vtis:
                part_stem = vti_path.stem
                cases.append({
                    "case_id": part_stem,
                    "output_name": part_stem,
                    "dataset_name": dataset_name,
                    "dataset_dims_xyz": dataset_dims,
                    "dataset_dtype": dataset_dtype,
                    "input_path": str(vti_path),
                    "input_name": vti_path.name,
                    "input_stem": _input_stem(str(vti_path)),
                    "input_reader": "vti",
                })
        return cases

    raw_files = sorted(path.resolve() for path in root.rglob("*.raw") if path.is_file())
    for raw_path in raw_files:
        dataset_name = raw_path.stem
        if selected_names is not None and dataset_name not in selected_names:
            continue
        dataset_id = _parse_dataset_id(dataset_name)
        x, y, z = dataset_id["dims_xyz"]
        cases.append({
            "case_id": dataset_name,
            "output_name": dataset_name,
            "dataset_name": dataset_name,
            "dataset_dims_xyz": [x, y, z],
            "dataset_dtype": dataset_id["dtype"],
            "input_path": str(raw_path),
            "input_name": raw_path.name,
            "input_stem": _input_stem(str(raw_path)),
            "input_reader": "raw",
            "io_override": {
                "shape_zyx": [z, y, x],
                "dtype": dataset_id["dtype"],
                "order": "zyx",
                "spacing": [1.0, 1.0, 1.0],
            },
        })
    return cases


def assign_cases_to_batches(
    cases: List[Dict[str, str]],
    batch_size: int = 100,
    shuffle: bool = True,
    seed: int | None = 0,
    batch_prefix: str = "batch_",
) -> List[Dict[str, str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    assigned = [dict(case) for case in cases]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(assigned)

    total_batches = (len(assigned) + batch_size - 1) // batch_size if assigned else 0
    width = max(4, len(str(total_batches or 1)))
    for index, case in enumerate(assigned):
        batch_index = (index // batch_size) + 1
        case["batch_index"] = batch_index
        case["batch_id"] = f"{batch_prefix}{batch_index:0{width}d}"
        case["batch_case_index"] = (index % batch_size) + 1
        case["batch_size"] = batch_size
        case["batch_shuffle"] = shuffle
        case["batch_seed"] = seed
    return assigned


def _normalize_cmap_pool(pool: Any) -> List[str]:
    if pool is None:
        return list(DEFAULT_CMAP_POOL)
    if isinstance(pool, list):
        return [str(x).strip() for x in pool if str(x).strip()]
    if isinstance(pool, str):
        return [t.strip() for t in re.split(r"[\n,]+", pool) if t.strip()]
    raise ValueError("render.cmaps_pool must be a list of names or a comma/newline-separated string")


def _rng_for_case(global_seed: Optional[int], case_id: str) -> random.Random:
    s = int(global_seed) if global_seed is not None else 0
    for ch in case_id:
        s = (s * 1315423911 + ord(ch)) & 0xFFFFFFFF
    return random.Random(s)


def _apply_random_cmaps(render_cfg: Dict[str, Any], case_info: Dict[str, str]) -> None:
    """若 render.cmaps_random 为 true，则为每个 TF band 从池中随机选配色（每病例可复现）。"""
    flag = bool(render_cfg.pop("cmaps_random", False))
    pool_raw = render_cfg.pop("cmaps_pool", None)
    seed_override = render_cfg.pop("cmaps_seed", None)
    if not flag:
        return
    pool = _normalize_cmap_pool(pool_raw)
    seed = seed_override if seed_override is not None else case_info.get("batch_seed")
    band_count = int(render_cfg.get("band_count", 10))
    rng = _rng_for_case(seed, case_info["case_id"])
    render_cfg["cmaps"] = [rng.choice(pool) for _ in range(band_count)]


def _render_template_values(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormatDict(context))
    if isinstance(value, list):
        return [_render_template_values(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_template_values(item, context) for key, item in value.items()}
    return value


def _resolve_case_output_dirs(case_info: Dict[str, Any], output_root: str) -> tuple[str, str]:
    batch_id = case_info.get("batch_id")
    output_name = case_info.get("output_name") or case_info["case_id"]
    if batch_id:
        case_output_dir = os.path.abspath(os.path.join(output_root, batch_id, output_name))
        batch_output_dir = os.path.abspath(os.path.join(output_root, batch_id))
    else:
        case_output_dir = os.path.abspath(os.path.join(output_root, output_name))
        batch_output_dir = os.path.abspath(output_root)
    return case_output_dir, batch_output_dir


def prepare_case_config_data(base_config_data: Dict[str, Any], case_info: Dict[str, Any], output_root: str, override_paths: bool = True) -> Dict[str, Any]:
    case_output_dir, batch_output_dir = _resolve_case_output_dirs(case_info, output_root)
    context = dict(case_info)
    context["output_dir"] = case_output_dir
    context["batch_output_dir"] = batch_output_dir
    context["canonical_vti_path"] = os.path.join(case_output_dir, f"{case_info['input_stem']}_canonical.vti")
    context["tiles_dir"] = os.path.join(case_output_dir, "tiles")
    context["export_path"] = case_output_dir

    config_data = _render_template_values(copy.deepcopy(base_config_data), context)
    config_data.setdefault("io", {})
    config_data["io"]["path"] = case_info["input_path"]
    if case_info.get("input_reader"):
        config_data["io"]["reader"] = case_info["input_reader"]
    io_override = case_info.get("io_override")
    if isinstance(io_override, dict):
        config_data["io"].update(io_override)
    io_tiling = config_data["io"].get("tiling") or config_data["io"].get("tile")
    if isinstance(io_tiling, dict) and (io_tiling.get("write_tiles") or io_tiling.get("output_dir")):
        io_tiling.setdefault("output_dir", os.path.join(case_output_dir, "input_tiles"))

    for stage in config_data.get("preprocess", []):
        if stage.get("name") != "canonicalize":
            continue
        if override_paths or not stage.get("vti_path"):
            stage["write_vti"] = True
            stage["vti_path"] = context["canonical_vti_path"]

    render_cfg = config_data.get("render")
    if isinstance(render_cfg, dict) and (override_paths or not render_cfg.get("path")):
        render_cfg["path"] = case_output_dir
        qc_cfg = render_cfg.get("qc")
        if isinstance(qc_cfg, dict):
            qc_cfg.setdefault("report_json", os.path.join(case_output_dir, "render_qc.json"))
            qc_cfg.setdefault("report_md", os.path.join(case_output_dir, "render_qc.md"))
        _apply_random_cmaps(render_cfg, case_info)
    if isinstance(render_cfg, dict) and not render_cfg.get("renderer"):
        render_cfg["renderer"] = "pv_engine"

    export_cfg = config_data.get("export")
    if isinstance(export_cfg, dict) and (override_paths or not export_cfg.get("path")):
        export_cfg["path"] = context["export_path"]

    # Backward compatibility: historical dataset batch configs may omit sampling.name.
    sampling_cfg = config_data.get("sampling")
    if isinstance(sampling_cfg, dict) and not sampling_cfg.get("name"):
        sampling_cfg["name"] = "opacity"
    return config_data


def _iter_tf_dirs(case_output_dir: str) -> Iterable[str]:
    if not os.path.isdir(case_output_dir):
        return
    for name in sorted(os.listdir(case_output_dir)):
        if not name.startswith("TF"):
            continue
        path = os.path.join(case_output_dir, name)
        if os.path.isdir(path):
            yield path


def _is_tf_dir_complete(tf_dir: str) -> bool:
    assert os.path.isdir(tf_dir), f"TF dir not found: {tf_dir}"
    has_tf_config = os.path.isfile(os.path.join(tf_dir, "tf_config.json"))
    has_metadata = os.path.isfile(os.path.join(tf_dir, "metadata.json"))
    has_split_json = any(
        os.path.isfile(os.path.join(tf_dir, name))
        for name in ("transforms_train.json", "transforms_test.json", "transforms_val.json")
    )
    has_train_png = os.path.isdir(os.path.join(tf_dir, "train")) and any(
        name.endswith(".png") for name in os.listdir(os.path.join(tf_dir, "train"))
    )
    has_test_png = os.path.isdir(os.path.join(tf_dir, "test")) and any(
        name.endswith(".png") for name in os.listdir(os.path.join(tf_dir, "test"))
    )
    has_images = has_train_png or has_test_png
    has_points = os.path.isfile(os.path.join(tf_dir, "points.ply")) or os.path.isfile(
        os.path.join(tf_dir, "point_cloud", "point_cloud_normalized.ply")
    )
    return has_tf_config and has_metadata and has_split_json and has_images and has_points


def _is_case_output_complete(case_output_dir: str) -> bool:
    if not os.path.isdir(case_output_dir):
        return False
    marker_path = os.path.join(case_output_dir, "_batch_done.json")
    if os.path.isfile(marker_path):
        with open(marker_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("tf_count", 0)) > 0
    tf_dirs = list(_iter_tf_dirs(case_output_dir))
    if not tf_dirs:
        return False
    return all(_is_tf_dir_complete(tf_dir) for tf_dir in tf_dirs)


def _ensure_pointcloud_normalized(tf_dir: str) -> None:
    src = os.path.join(tf_dir, "points.ply")
    if not os.path.isfile(src):
        return
    dst_dir = os.path.join(tf_dir, "point_cloud")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "point_cloud_normalized.ply")
    shutil.copyfile(src, dst)


def _write_case_done_marker(case_output_dir: str, tf_dirs: List[str]) -> None:
    payload = {
        "status": "done",
        "tf_count": len(tf_dirs),
        "tf_names": [os.path.basename(path) for path in sorted(tf_dirs)],
    }
    with open(os.path.join(case_output_dir, "_batch_done.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _tf_name_to_cmap(tf_name: str, cmaps: Any) -> Any:
    if not isinstance(cmaps, list):
        return None
    m = re.match(r"^TF(\d+)$", tf_name)
    if not m:
        return None
    index = int(m.group(1)) - 1
    if index < 0 or index >= len(cmaps):
        return None
    return cmaps[index]


def _write_tf_metadata(tf_dir: str, case_info: Dict[str, Any], render_params: Dict[str, Any]) -> None:
    metadata = {
        "volume_name": case_info.get("dataset_name") or case_info.get("case_id"),
        "volume_dims_xyz": case_info.get("dataset_dims_xyz"),
        "tf_cmap": _tf_name_to_cmap(os.path.basename(tf_dir), render_params.get("cmaps")),
        "tf_mode": render_params.get("tf_mode"),
        "band_count": render_params.get("band_count"),
        "hist_eq": bool(render_params.get("hist_eq", False)),
    }
    with open(os.path.join(tf_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _finalize_case_outputs(case_output_dir: str, case_info: Dict[str, Any], cfg: Config, context: Dict[str, Any]) -> None:
    sampled_tasks = context.get("sampled_tasks") or []
    sampled_tf_names = {
        task.get("tf_name")
        for task in sampled_tasks
        if isinstance(task, dict) and task.get("tf_name")
    }
    has_named_sampling_tasks = any(
        isinstance(task, dict) and task.get("tf_name")
        for task in (context.get("sampling_tasks") or [])
    )

    render_params = cfg.render.params if cfg.render and isinstance(cfg.render.params, dict) else {}
    kept_tf_dirs: List[str] = []
    for tf_dir in list(_iter_tf_dirs(case_output_dir)):
        tf_name = os.path.basename(tf_dir)
        if has_named_sampling_tasks and tf_name not in sampled_tf_names:
            shutil.rmtree(tf_dir)
            continue
        _ensure_pointcloud_normalized(tf_dir)
        if case_info.get("dataset_name"):
            _write_tf_metadata(tf_dir, case_info, render_params)
        kept_tf_dirs.append(tf_dir)
    if kept_tf_dirs:
        _write_case_done_marker(case_output_dir, kept_tf_dirs)


def _write_batch_reports(output_root: str, report: Dict[str, Any]) -> None:
    json_path = os.path.join(output_root, "batch_report.json")
    md_path = os.path.join(output_root, "batch_report.md")
    os.makedirs(os.path.abspath(output_root), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Batch Report",
        "",
        f"- total_cases: {report['total_cases']}",
        f"- total_batches: {report.get('total_batches', 0)}",
        f"- batch_size: {report.get('batch_size', '-')}",
        f"- shuffle: {report.get('shuffle', '-')}",
        f"- seed: {report.get('seed', '-')}",
        f"- succeeded: {report['succeeded']}",
        f"- skipped: {report.get('skipped', 0)}",
        f"- failed: {report['failed']}",
        "",
        "| Batch | Case | Status | Input | QC Failed TF | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        qc_failed = ",".join(item.get("failed_tf_names", [])) if item.get("failed_tf_names") else "-"
        error = item.get("error", "-")
        lines.append(f"| {item.get('batch_id', '-')} | {item['case_id']} | {item['status']} | {item['input_name']} | {qc_failed} | {error} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    report["report_json"] = os.path.abspath(json_path)
    report["report_md"] = os.path.abspath(md_path)


def run_batch(
    config_path: str,
    raw_root: str = "raw",
    output_root: str = "outputs",
    case_glob: str = "s*",
    filename: str | None = None,
    continue_on_error: bool = True,
    override_paths: bool = True,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
    batch_index: int | None = None,
    input_mode: str = "case",
    datasets: str | None = None,
    datasets_root: str = "datasets_for_volume_3dgs",
    volumes_root: str = "volumes",
    vti_cache_root: str = "vti_cache",
    dataset_source_root: str | None = None,
    max_vti_parts: int | None = None,
    skip_existing: bool = False,
    dataset_vti_only: bool = False,
) -> Dict[str, Any]:
    base_config_data = load_config_data(config_path)
    raw_batch_cfg = base_config_data.get("batch")
    batch_cfg = raw_batch_cfg or {}
    has_explicit_batch_cfg = isinstance(raw_batch_cfg, dict) and len(raw_batch_cfg) > 0
    mode = str(input_mode).strip().lower()
    if mode not in {"case", "dataset"}:
        raise ValueError(f"Unsupported input_mode: {input_mode} (expected 'case' or 'dataset')")
    if mode == "case":
        reader = base_config_data.get("io", {}).get("reader", "nii")
        cases = discover_case_inputs(raw_root, reader=reader, case_glob=case_glob, filename=filename)
    else:
        if dataset_source_root:
            cases = discover_dataset_inputs_from_source_root(
                source_root=dataset_source_root,
                datasets=datasets,
                max_vti_parts=max_vti_parts,
            )
        else:
            cases = discover_dataset_inputs(
                datasets=datasets,
                datasets_root=datasets_root,
                volumes_root=volumes_root,
                vti_cache_root=vti_cache_root,
                max_vti_parts=max_vti_parts,
                vti_only=dataset_vti_only,
            )
    dataset_plain_layout = (
        mode == "dataset"
        and not has_explicit_batch_cfg
        and batch_size is None
        and shuffle is None
        and seed is None
        and batch_index is None
    )
    resolved_batch_size = int(batch_size if batch_size is not None else batch_cfg.get("size", 100))
    resolved_shuffle = bool(shuffle if shuffle is not None else batch_cfg.get("shuffle", (False if dataset_plain_layout else True)))
    resolved_seed = seed if seed is not None else batch_cfg.get("seed", 0)
    selected_batch_index = batch_index if batch_index is not None else batch_cfg.get("batch_index")
    batch_prefix = str(batch_cfg.get("prefix", "batch_"))

    if dataset_plain_layout:
        assigned = [dict(case) for case in cases]
        if resolved_shuffle:
            random.Random(resolved_seed).shuffle(assigned)
        cases = assigned
        total_batches = 0
    else:
        cases = assign_cases_to_batches(
            cases,
            batch_size=resolved_batch_size,
            shuffle=resolved_shuffle,
            seed=resolved_seed,
            batch_prefix=batch_prefix,
        )
        total_batches = max((case["batch_index"] for case in cases), default=0)
    if selected_batch_index is not None:
        selected_batch_index = int(selected_batch_index)
        cases = [case for case in cases if case["batch_index"] == selected_batch_index]
        if not cases:
            raise ValueError(f"Requested batch_index={selected_batch_index} but no cases were assigned to that batch")

    report: Dict[str, Any] = {
        "config_path": os.path.abspath(config_path),
        "raw_root": os.path.abspath(raw_root),
        "output_root": os.path.abspath(output_root),
        "input_mode": mode,
        "batch_size": resolved_batch_size,
        "shuffle": resolved_shuffle,
        "seed": resolved_seed,
        "requested_batch_index": selected_batch_index,
        "total_batches": total_batches,
        "total_cases": len(cases),
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        "cases": [],
    }

    for case_info in cases:
        case_output_dir, _ = _resolve_case_output_dirs(case_info, output_root)
        if skip_existing and _is_case_output_complete(case_output_dir):
            report["cases"].append({
                "batch_id": case_info.get("batch_id"),
                "batch_index": case_info.get("batch_index"),
                "batch_case_index": case_info.get("batch_case_index"),
                "case_id": case_info["case_id"],
                "input_name": case_info["input_name"],
                "input_path": case_info["input_path"],
                "status": "skipped",
                "case_output_dir": case_output_dir,
            })
            report["skipped"] += 1
            continue
        config_data = prepare_case_config_data(base_config_data, case_info, output_root=output_root, override_paths=override_paths)
        cfg = Config.from_dict(config_data)
        entry = {
            "batch_id": case_info.get("batch_id"),
            "batch_index": case_info.get("batch_index"),
            "batch_case_index": case_info.get("batch_case_index"),
            "case_id": case_info["case_id"],
            "input_name": case_info["input_name"],
            "input_path": case_info["input_path"],
            "case_output_dir": case_output_dir,
            "status": "pending",
        }
        try:
            context = run_pipeline_context(None, None, cfg)
            render_result = context.get("render_result") or {}
            qc_result = render_result.get("qc") or {}
            tf_cmaps = None
            if cfg.render and isinstance(cfg.render.params, dict):
                tf_cmaps = cfg.render.params.get("cmaps")
            entry.update({
                "status": "ok",
                "canonical_vti_path": context.get("canonical_vti_path"),
                "written_paths": context.get("written_paths", []),
                "failed_tf_names": qc_result.get("failed_tf_names", []),
                "passed_tf_names": qc_result.get("passed_tf_names", []),
                "qc_report_json": qc_result.get("report_json"),
                "qc_report_md": qc_result.get("report_md"),
                "tf_cmaps": tf_cmaps,
            })
            _finalize_case_outputs(case_output_dir, case_info, cfg, context)
            report["succeeded"] += 1
        except Exception as exc:
            entry.update({
                "status": "failed",
                "error": str(exc),
            })
            report["failed"] += 1
            if not continue_on_error:
                report["cases"].append(entry)
                _write_batch_reports(output_root, report)
                raise
        report["cases"].append(entry)

    _write_batch_reports(output_root, report)
    return report
