#!/usr/bin/env python3
"""
从每个 case 的 render_qc.json 中选择 top-K TF，并把选中的 TF 软链接到新目录。

默认策略：
1. 优先保留 status=pass 的 TF；
2. 若 pass 数量不足 K，则从 fail 中按视觉代理指标回填；
3. fail 里优先选择不是 too_sparse / too_dark / low_contrast 的 TF；
4. 在同一严重度内，优先保留 max_detail_over_opacity / detail_over_opacity 更高的 TF。

示例：
  python scripts/select_top_tf_from_render_qc.py \
    --dataset-root outputs/totalseg3d_qc_remake_v1 \
    --top-k 3 \
    --clean-output-root
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from vol2splat.config import load_config_data
except Exception:
    load_config_data = None


SEVERE_REASON_WEIGHTS: dict[str, int] = {
    "too_sparse": 2,
    "too_dark": 2,
    "low_contrast": 1,
    "no_images": 4,
    "segmentation_failed": 3,
}


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _severity_score(reason_codes: list[str]) -> int:
    return sum(SEVERE_REASON_WEIGHTS.get(reason, 0) for reason in reason_codes)


def _rank_key(item: dict[str, Any]) -> tuple:
    metrics = item.get("metrics") or {}
    reason_codes = list(item.get("reason_codes") or [])
    status = str(item.get("status") or "")
    rescue_codes = list(item.get("rescue_codes") or [])
    rescued = 0 if rescue_codes else 1
    severe_score = _severity_score(reason_codes)
    # Lower tuple values rank earlier.
    return (
        0 if status == "pass" else 1,
        rescued,
        severe_score,
        -float(metrics.get("max_detail_over_opacity", 0.0)),
        -float(metrics.get("detail_over_opacity", 0.0)),
        -float(metrics.get("max_masked_fft_high_freq_ratio", 0.0)),
        -float(metrics.get("masked_fft_high_freq_ratio", 0.0)),
        float(metrics.get("foreground_alpha_mean", 1.0)),
        -float(metrics.get("intensity_std", 0.0)),
        -float(metrics.get("mean_intensity", 0.0)),
        -float(metrics.get("nonzero_ratio", 0.0)),
        str(item.get("tf_name") or ""),
    )


def select_top_tf_items(report: dict[str, Any], top_k: int = 3) -> list[dict[str, Any]]:
    items = list(report.get("items") or [])
    ranked = sorted(items, key=_rank_key)
    if top_k <= 0:
        return []
    return ranked[:top_k]


def _discover_case_reports(dataset_root: Path, output_root: Path | None = None) -> list[Path]:
    reports: list[Path] = []
    output_root = output_root.resolve() if output_root is not None else None
    for report_path in sorted(dataset_root.rglob("render_qc.json")):
        if output_root is not None and output_root in report_path.parents:
            continue
        reports.append(report_path.resolve())
    return reports


def build_top_tf_symlinks(
    dataset_root: Path,
    output_root: Path,
    top_k: int = 3,
    clean_output_root: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if clean_output_root and output_root.exists():
        _remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    selected_links: list[dict[str, str]] = []

    for report_path in _discover_case_reports(dataset_root, output_root=output_root):
        case_dir = report_path.parent.resolve()
        rel_case_dir = case_dir.relative_to(dataset_root)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_items = select_top_tf_items(report, top_k=top_k)

        dst_case_dir = output_root / rel_case_dir
        dst_case_dir.mkdir(parents=True, exist_ok=True)

        for name in ("render_qc.json", "render_qc.md"):
            src = case_dir / name
            if src.exists():
                dst = dst_case_dir / name
                if dst.exists() or dst.is_symlink():
                    _remove_path(dst)
                os.symlink(src, dst)

        selected_names: list[str] = []
        ranking: list[dict[str, Any]] = []
        for item in selected_items:
            tf_name = str(item["tf_name"])
            src_tf_dir = case_dir / tf_name
            if not src_tf_dir.is_dir():
                continue
            dst_tf_link = dst_case_dir / tf_name
            if dst_tf_link.exists() or dst_tf_link.is_symlink():
                _remove_path(dst_tf_link)
            os.symlink(src_tf_dir, dst_tf_link)
            selected_links.append(
                {
                    "case_dir": str(case_dir),
                    "case_relative_dir": str(rel_case_dir).replace("\\", "/"),
                    "tf_name": tf_name,
                    "src": str(src_tf_dir),
                    "dst": str(dst_tf_link),
                }
            )
            selected_names.append(tf_name)
            ranking.append(
                {
                    "tf_name": tf_name,
                    "status": item.get("status"),
                    "reason_codes": item.get("reason_codes") or [],
                    "rescue_codes": item.get("rescue_codes") or [],
                    "metrics": {
                        key: (item.get("metrics") or {}).get(key)
                        for key in (
                            "nonzero_ratio",
                            "mean_intensity",
                            "intensity_std",
                            "masked_fft_high_freq_ratio",
                            "detail_over_opacity",
                            "max_masked_fft_high_freq_ratio",
                            "max_detail_over_opacity",
                            "foreground_alpha_mean",
                        )
                    },
                }
            )

        cases.append(
            {
                "case_dir": str(case_dir),
                "case_relative_dir": str(rel_case_dir).replace("\\", "/"),
                "selected_tf_names": selected_names,
                "selected": ranking,
                "total_tf": len(report.get("items") or []),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "top_k": int(top_k),
        "num_cases": len(cases),
        "num_links": len(selected_links),
        "cases": cases,
        "links": selected_links,
    }
    report_path = output_root / "top_tf_selection_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["report_json"] = str(report_path)
    return summary


def _load_top_tf_selection_cfg(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    if load_config_data is None:
        raise SystemExit("Unable to import vol2splat.config.load_config_data; run this script from the repo environment.")
    config_data = load_config_data(str(config_path))
    return ((config_data.get("selection") or {}).get("top_tf") or {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None, help="可选配置文件；默认从 selection.top_tf / batch.output_root 读取")
    ap.add_argument("--dataset-root", type=Path, default=None, help="包含 case 渲染目录的根目录")
    ap.add_argument("--output-root", type=Path, default=None, help="输出软链接根目录；默认 <dataset-root>/top_tf")
    ap.add_argument("--top-k", type=int, default=None, help="每个 case 保留的 TF 数量")
    ap.add_argument("--clean-output-root", action="store_true", help="若输出目录已存在，则先删除再重建")
    args = ap.parse_args()

    config_path = args.config.expanduser().resolve() if args.config else None
    selection_cfg = _load_top_tf_selection_cfg(config_path)
    batch_cfg: dict[str, Any] = {}
    if config_path is not None:
        if load_config_data is None:
            raise SystemExit("Unable to import vol2splat.config.load_config_data; run this script from the repo environment.")
        batch_cfg = (load_config_data(str(config_path)).get("batch") or {})

    resolved_dataset_root = args.dataset_root or selection_cfg.get("dataset_root") or batch_cfg.get("output_root")
    if resolved_dataset_root is None:
        raise SystemExit("dataset-root is required unless config provides selection.top_tf.dataset_root or batch.output_root")

    dataset_root = Path(resolved_dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root is not a directory: {dataset_root}")

    resolved_output_root = args.output_root or selection_cfg.get("output_root")
    output_root = Path(resolved_output_root).expanduser().resolve() if resolved_output_root else (dataset_root / "top_tf")
    top_k = int(args.top_k if args.top_k is not None else selection_cfg.get("top_k", 3))
    report = build_top_tf_symlinks(
        dataset_root=dataset_root,
        output_root=output_root,
        top_k=top_k,
        clean_output_root=bool(args.clean_output_root),
    )
    print(
        f"selected top TFs: cases={report['num_cases']} "
        f"links={report['num_links']} top_k={report['top_k']} output={report['output_root']}"
    )
    print(f"report: {report['report_json']}")


if __name__ == "__main__":
    main()
