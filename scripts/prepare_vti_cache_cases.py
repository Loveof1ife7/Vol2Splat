#!/usr/bin/env python3
"""
将 vti_cache 下的所有 .vti 映射为“每个文件一个 case”的目录树。

输入:
  raw_sim/vti_cache/<dataset_dir>/*.vti

输出:
  raw_sim/vti_cache_cases/<vti_stem>/input.vti -> 原始 .vti

示例:
  python scripts/prepare_vti_cache_cases.py \
    --source-root raw_sim/vti_cache \
    --output-root raw_sim/vti_cache_cases \
    --clean-output-root
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def build_vti_case_links(
    source_root: Path,
    output_root: Path,
    link_name: str = "input.vti",
    clean_output_root: bool = False,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    if clean_output_root and output_root.exists():
        _remove_path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, str]] = []
    seen_case_ids: set[str] = set()
    for vti_path in sorted(source_root.rglob("*.vti")):
        if not vti_path.is_file():
            continue
        case_id = vti_path.stem
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id derived from stem {case_id!r}: {vti_path}")
        seen_case_ids.add(case_id)

        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        dst_link = case_dir / link_name
        if dst_link.exists() or dst_link.is_symlink():
            _remove_path(dst_link)
        os.symlink(vti_path.resolve(), dst_link)
        cases.append(
            {
                "case_id": case_id,
                "source_dataset_dir": str(vti_path.parent.relative_to(source_root)).replace("\\", "/"),
                "source_vti": str(vti_path.resolve()),
                "case_dir": str(case_dir.resolve()),
                "linked_input": str(dst_link.resolve()),
            }
        )

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "link_name": link_name,
        "total_cases": len(cases),
        "cases": cases,
    }
    report_path = output_root / "_vti_case_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True, help="源 vti_cache 根目录")
    ap.add_argument("--output-root", type=Path, required=True, help="输出 case 根目录")
    ap.add_argument("--link-name", type=str, default="input.vti", help="每个 case 内的输入文件名")
    ap.add_argument("--clean-output-root", action="store_true", help="若输出目录已存在，则先删除再重建")
    args = ap.parse_args()

    report = build_vti_case_links(
        source_root=args.source_root,
        output_root=args.output_root,
        link_name=args.link_name,
        clean_output_root=bool(args.clean_output_root),
    )
    print(
        f"prepared vti cache cases: total_cases={report['total_cases']} "
        f"output_root={report['output_root']}"
    )
    print(f"manifest: {report['report_json']}")


if __name__ == "__main__":
    main()
