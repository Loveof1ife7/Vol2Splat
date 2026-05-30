#!/usr/bin/env python3
"""
根据数据集级 QC summary，把 passed TF 以软链接方式镜像到一个新的 qc/ 目录树里。

默认结构：

  <dataset-root>/qc/<batch_id>/<case_id>/<TFxx> -> <dataset-root>/<batch_id>/<case_id>/<TFxx>

示例：
  python scripts/link_passed_tf_from_qc.py \
    --dataset-root outputs/overfitting_usage \
    --clean-output-root
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _load_dataset_summary(summary_json: Path) -> dict[str, Any]:
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    if not isinstance(data.get("cases"), list):
        raise SystemExit(f"summary missing cases list: {summary_json}")
    return data


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def build_passed_tf_symlinks(
    summary_json: Path,
    output_root: Path,
    clean_output_root: bool = False,
) -> dict[str, Any]:
    summary_json = summary_json.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    dataset_summary = _load_dataset_summary(summary_json)

    if clean_output_root and output_root.exists():
        _remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    created_links: list[dict[str, str]] = []
    num_cases = 0
    num_tf = 0

    for case in dataset_summary["cases"]:
        batch_id = str(case["batch_id"])
        case_id = str(case["case_id"])
        case_dir = Path(case["case_dir"]).resolve()
        passed_tf_names = list(case.get("passed_tf_names") or [])
        num_cases += 1

        dst_case_dir = output_root / batch_id / case_id
        dst_case_dir.mkdir(parents=True, exist_ok=True)

        for tf_name in passed_tf_names:
            src_tf_dir = case_dir / tf_name
            if not src_tf_dir.is_dir():
                raise SystemExit(f"passed TF source dir not found: {src_tf_dir}")

            dst_tf_link = dst_case_dir / tf_name
            if dst_tf_link.exists() or dst_tf_link.is_symlink():
                _remove_path(dst_tf_link)

            os.symlink(src_tf_dir, dst_tf_link)
            created_links.append(
                {
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "tf_name": tf_name,
                    "src": str(src_tf_dir),
                    "dst": str(dst_tf_link),
                }
            )
            num_tf += 1

    report = {
        "summary_json": str(summary_json),
        "output_root": str(output_root),
        "num_cases": num_cases,
        "num_passed_tf_links": num_tf,
        "links": created_links,
    }
    report_path = output_root / "passed_tf_links_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="含 batch_0001/batch_0002/... 的数据集根目录",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="数据集级 qc summary json；默认 <dataset-root>/qc_lowfreq_summary.json",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出软链接根目录；默认 <dataset-root>/qc",
    )
    ap.add_argument(
        "--clean-output-root",
        action="store_true",
        help="若输出目录已存在，则先整棵删除再重建",
    )
    args = ap.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root is not a directory: {dataset_root}")

    summary_json = args.summary_json or (dataset_root / "qc_lowfreq_summary.json")
    output_root = args.output_root or (dataset_root / "qc")

    report = build_passed_tf_symlinks(
        summary_json=summary_json,
        output_root=output_root,
        clean_output_root=bool(args.clean_output_root),
    )
    print(
        f"created passed TF symlinks: cases={report['num_cases']} "
        f"passed_tf={report['num_passed_tf_links']} output={report['output_root']}"
    )
    print(f"report: {report['report_json']}")


if __name__ == "__main__":
    main()
