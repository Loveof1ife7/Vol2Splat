#!/usr/bin/env python3
"""
为按 colormap 拆分的数据集准备 canonical.vti。

设计目标：
1. 选择一个 anchor colormap 数据集，给它的每个 case 放一份真实 canonical.vti
2. anchor 数据集下每个 TFxx/scene 目录中的 canonical.vti 使用相对软链接 -> ../canonical.vti
3. 其它 colormap 数据集的 case 根目录 canonical.vti 相对软链接到 anchor 数据集对应 case
4. 其它 colormap 数据集的 TFxx/scene 目录 canonical.vti 相对软链接 -> ../canonical.vti

这样迁移到其它服务器时，只要整棵 by_cmap 输出目录一起搬走，内部软链接仍然有效。

示例：
  python scripts/prepare_cmap_variant_canonical_vti.py \
    --canonical-root raw_sim/openscivis_canonical_vti \
    --by-cmap-root outputs/openscivis_canonical_render_only_by_cmap \
    --anchor-dataset miranda_render_only_cool_to_warm_extended
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
    if path.exists():
        raise SystemExit(f"destination exists and is not a file/symlink: {path}")


def _ensure_relative_symlink(src: Path, dst: Path) -> None:
    _remove_path(dst)
    rel_src = os.path.relpath(str(src), start=str(dst.parent))
    os.symlink(rel_src, dst)


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [item.strip() for item in value.split(",") if item.strip()]
    return out or None


def _resolve_source_vti(case_id: str, canonical_root: Path) -> Path:
    case_root = canonical_root / case_id
    if not case_root.is_dir():
        raise SystemExit(f"canonical case dir not found: {case_root}")
    preferred = [
        case_root / f"{case_id}_canonical.vti",
        case_root / "canonical.vti",
    ]
    for path in preferred:
        if path.is_file():
            return path.resolve()
    candidates = sorted(path.resolve() for path in case_root.glob("*.vti") if path.is_file())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"no VTI found under canonical case dir: {case_root}")
    raise SystemExit(f"multiple VTI files found under canonical case dir, cannot choose uniquely: {case_root}")


def _iter_case_dirs(dataset_root: Path, case_ids: list[str] | None) -> list[Path]:
    if case_ids:
        case_dirs = [dataset_root / case_id for case_id in case_ids]
        missing = [path for path in case_dirs if not path.is_dir()]
        if missing:
            raise SystemExit(f"missing case dirs under {dataset_root}: {', '.join(str(path) for path in missing)}")
        return case_dirs
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def _iter_tf_dirs(case_dir: Path, tf_glob: str) -> list[Path]:
    return sorted(path for path in case_dir.glob(tf_glob) if path.is_dir())


def _copy_anchor_case_vti(src_vti: Path, dst_vti: Path) -> None:
    dst_vti.parent.mkdir(parents=True, exist_ok=True)
    if dst_vti.is_symlink():
        dst_vti.unlink()
    if dst_vti.exists():
        if dst_vti.samefile(src_vti):
            return
        dst_vti.unlink()
    shutil.copy2(src_vti, dst_vti)


def build_cmap_variant_canonical_vti(
    canonical_root: Path,
    by_cmap_root: Path,
    anchor_dataset: str,
    case_ids: list[str] | None = None,
    dataset_names: list[str] | None = None,
    tf_glob: str = "TF*",
    link_name: str = "canonical.vti",
) -> dict[str, Any]:
    canonical_root = canonical_root.expanduser().resolve()
    by_cmap_root = by_cmap_root.expanduser().resolve()
    if not canonical_root.is_dir():
        raise SystemExit(f"canonical root is not a directory: {canonical_root}")
    if not by_cmap_root.is_dir():
        raise SystemExit(f"by-cmap root is not a directory: {by_cmap_root}")

    anchor_root = by_cmap_root / anchor_dataset
    if not anchor_root.is_dir():
        raise SystemExit(f"anchor dataset dir not found: {anchor_root}")

    candidate_dataset_roots = sorted(path for path in by_cmap_root.iterdir() if path.is_dir())
    if dataset_names:
        selected = []
        missing = []
        for name in dataset_names:
            path = by_cmap_root / name
            if path.is_dir():
                selected.append(path)
            else:
                missing.append(name)
        if missing:
            raise SystemExit(f"dataset dirs not found under {by_cmap_root}: {', '.join(missing)}")
        candidate_dataset_roots = selected
    if anchor_root not in candidate_dataset_roots:
        candidate_dataset_roots.insert(0, anchor_root)

    copied_cases: list[dict[str, str]] = []
    linked_case_roots: list[dict[str, str]] = []
    linked_tf_dirs: list[dict[str, str]] = []

    anchor_case_dirs = _iter_case_dirs(anchor_root, case_ids=case_ids)
    for case_dir in anchor_case_dirs:
        case_id = case_dir.name
        src_vti = _resolve_source_vti(case_id, canonical_root=canonical_root)
        anchor_case_vti = case_dir / link_name
        _copy_anchor_case_vti(src_vti, anchor_case_vti)
        copied_cases.append({"case_id": case_id, "src": str(src_vti), "dst": str(anchor_case_vti)})

        for tf_dir in _iter_tf_dirs(case_dir, tf_glob=tf_glob):
            dst_vti = tf_dir / link_name
            _ensure_relative_symlink(anchor_case_vti, dst_vti)
            linked_tf_dirs.append({"dataset": anchor_dataset, "case_id": case_id, "src": str(anchor_case_vti), "dst": str(dst_vti)})

    for dataset_root in candidate_dataset_roots:
        if dataset_root == anchor_root:
            continue
        for case_dir in _iter_case_dirs(dataset_root, case_ids=case_ids):
            case_id = case_dir.name
            anchor_case_vti = anchor_root / case_id / link_name
            if not anchor_case_vti.is_file():
                raise SystemExit(f"anchor canonical.vti not found for case {case_id}: {anchor_case_vti}")

            dst_case_vti = case_dir / link_name
            _ensure_relative_symlink(anchor_case_vti, dst_case_vti)
            linked_case_roots.append({"dataset": dataset_root.name, "case_id": case_id, "src": str(anchor_case_vti), "dst": str(dst_case_vti)})

            for tf_dir in _iter_tf_dirs(case_dir, tf_glob=tf_glob):
                dst_vti = tf_dir / link_name
                _ensure_relative_symlink(dst_case_vti, dst_vti)
                linked_tf_dirs.append({"dataset": dataset_root.name, "case_id": case_id, "src": str(dst_case_vti), "dst": str(dst_vti)})

    report = {
        "canonical_root": str(canonical_root),
        "by_cmap_root": str(by_cmap_root),
        "anchor_dataset": anchor_dataset,
        "dataset_names": [path.name for path in candidate_dataset_roots],
        "case_ids": case_ids or [],
        "tf_glob": tf_glob,
        "link_name": link_name,
        "num_anchor_case_copies": len(copied_cases),
        "num_case_root_links": len(linked_case_roots),
        "num_tf_dir_links": len(linked_tf_dirs),
        "anchor_case_copies": copied_cases,
        "case_root_links": linked_case_roots,
        "tf_dir_links": linked_tf_dirs,
    }
    report_path = by_cmap_root / "canonical_vti_variant_prep_summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical-root", type=Path, default=Path("raw_sim/openscivis_canonical_vti"))
    ap.add_argument("--by-cmap-root", type=Path, default=Path("outputs/openscivis_canonical_render_only_by_cmap"))
    ap.add_argument("--anchor-dataset", type=str, default="miranda_render_only_cool_to_warm_extended")
    ap.add_argument("--case-ids", type=str, default=None, help="optional comma-separated case ids")
    ap.add_argument("--dataset-names", type=str, default=None, help="optional comma-separated dataset dir names under by-cmap-root")
    ap.add_argument("--tf-glob", type=str, default="TF*")
    ap.add_argument("--link-name", type=str, default="canonical.vti")
    args = ap.parse_args()

    report = build_cmap_variant_canonical_vti(
        canonical_root=args.canonical_root,
        by_cmap_root=args.by_cmap_root,
        anchor_dataset=str(args.anchor_dataset),
        case_ids=_parse_csv_list(args.case_ids),
        dataset_names=_parse_csv_list(args.dataset_names),
        tf_glob=str(args.tf_glob),
        link_name=str(args.link_name),
    )
    print(
        f"prepared canonical VTI for cmap variants: "
        f"anchor_copies={report['num_anchor_case_copies']} "
        f"case_links={report['num_case_root_links']} "
        f"tf_links={report['num_tf_dir_links']} "
        f"root={report['by_cmap_root']}"
    )
    print(f"report: {report['report_json']}")


if __name__ == "__main__":
    main()
