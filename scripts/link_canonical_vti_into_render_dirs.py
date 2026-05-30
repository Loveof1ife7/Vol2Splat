#!/usr/bin/env python3
"""
为 render 输出目录补充 canonical.vti 软链接。

默认行为：
1. 遍历 render_root 下的每个 case 目录
2. 从 canonical_root/<case_id>/ 找到对应的 canonical VTI
3. 软链接到：
   - <render_root>/<case_id>/canonical.vti
   - <render_root>/<case_id>/<TFxx>/canonical.vti

示例：
  python scripts/link_canonical_vti_into_render_dirs.py \
    --canonical-root raw_sim/openscivis_canonical_vti \
    --render-root outputs/openscivis_canonical_render_only
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _remove_existing_link_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        raise SystemExit(f"destination exists and is not a file/symlink: {path}")


def _parse_case_ids(case_ids: str | None) -> list[str] | None:
    if case_ids is None:
        return None
    parsed = [item.strip() for item in case_ids.split(",") if item.strip()]
    return parsed or None


def _resolve_source_vti(case_id: str, canonical_root: Path) -> Path:
    case_root = canonical_root / case_id
    if not case_root.is_dir():
        raise SystemExit(f"canonical case dir not found: {case_root}")

    preferred_paths = [
        case_root / f"{case_id}_canonical.vti",
        case_root / "canonical.vti",
    ]
    for path in preferred_paths:
        if path.is_file():
            return path.resolve()

    candidates = sorted(path.resolve() for path in case_root.glob("*.vti") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"no VTI found under canonical case dir: {case_root}")
    raise SystemExit(f"multiple VTI files found under canonical case dir, cannot choose uniquely: {case_root}")


def _iter_render_case_dirs(render_root: Path, case_ids: list[str] | None) -> list[Path]:
    if case_ids:
        case_dirs = [render_root / case_id for case_id in case_ids]
        missing = [path for path in case_dirs if not path.is_dir()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise SystemExit(f"render case dir not found: {missing_text}")
        return case_dirs
    return sorted(path for path in render_root.iterdir() if path.is_dir())


def _iter_link_destinations(case_dir: Path, scope: str, tf_glob: str, link_name: str) -> list[Path]:
    destinations: list[Path] = []
    if scope in {"case-root", "both"}:
        destinations.append(case_dir / link_name)
    if scope in {"tf-dirs", "both"}:
        destinations.extend(tf_dir / link_name for tf_dir in sorted(path for path in case_dir.glob(tf_glob) if path.is_dir()))
    return destinations


def build_canonical_vti_symlinks(
    canonical_root: Path,
    render_root: Path,
    scope: str = "both",
    tf_glob: str = "TF*",
    link_name: str = "canonical.vti",
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    canonical_root = canonical_root.expanduser().resolve()
    render_root = render_root.expanduser().resolve()
    if not canonical_root.is_dir():
        raise SystemExit(f"canonical root is not a directory: {canonical_root}")
    if not render_root.is_dir():
        raise SystemExit(f"render root is not a directory: {render_root}")

    linked_items: list[dict[str, str]] = []
    num_cases = 0

    for case_dir in _iter_render_case_dirs(render_root, case_ids=case_ids):
        case_id = case_dir.name
        src_vti = _resolve_source_vti(case_id, canonical_root=canonical_root)
        destinations = _iter_link_destinations(case_dir, scope=scope, tf_glob=tf_glob, link_name=link_name)
        num_cases += 1

        for dst_path in destinations:
            _remove_existing_link_target(dst_path)
            os.symlink(src_vti, dst_path)
            linked_items.append(
                {
                    "case_id": case_id,
                    "src": str(src_vti),
                    "dst": str(dst_path),
                }
            )

    report = {
        "canonical_root": str(canonical_root),
        "render_root": str(render_root),
        "scope": scope,
        "tf_glob": tf_glob,
        "link_name": link_name,
        "case_ids": case_ids or [],
        "num_cases": num_cases,
        "num_links": len(linked_items),
        "links": linked_items,
    }
    report_path = render_root / "canonical_vti_links_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("raw_sim/openscivis_canonical_vti"),
        help="canonical VTI 数据集根目录",
    )
    ap.add_argument(
        "--render-root",
        type=Path,
        default=Path("outputs/openscivis_canonical_render_only"),
        help="render 输出根目录",
    )
    ap.add_argument(
        "--scope",
        choices=("case-root", "tf-dirs", "both"),
        default="both",
        help="链接到 case 根目录、TF 子目录，或两者都链接",
    )
    ap.add_argument(
        "--tf-glob",
        type=str,
        default="TF*",
        help="TF 子目录 glob，仅在 scope 包含 tf-dirs 时生效",
    )
    ap.add_argument(
        "--link-name",
        type=str,
        default="canonical.vti",
        help="目标软链接文件名",
    )
    ap.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help="可选，逗号分隔的 case_id 子集",
    )
    args = ap.parse_args()

    report = build_canonical_vti_symlinks(
        canonical_root=args.canonical_root,
        render_root=args.render_root,
        scope=str(args.scope),
        tf_glob=str(args.tf_glob),
        link_name=str(args.link_name),
        case_ids=_parse_case_ids(args.case_ids),
    )
    print(
        f"linked canonical VTI: cases={report['num_cases']} "
        f"links={report['num_links']} scope={report['scope']} render_root={report['render_root']}"
    )
    print(f"report: {report['report_json']}")


if __name__ == "__main__":
    main()
