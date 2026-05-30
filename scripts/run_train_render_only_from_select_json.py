#!/usr/bin/env python3
"""
从 select_train_cases_near_test_ct_stats.py 的 --json-out 读取 case_id，
为每个训练病例在 raw/high_quality/medical/<id>/ 下软链测试池 TF（默认同目录随机源），
再调用 vol2splat batch（仅渲染配置、无点云）。

用法（在 Vol2Splat 仓库根目录）:
  python scripts/run_train_render_only_from_select_json.py \\
    --json-out outputs/train_select_near_test_stats.json \\
    --output-root outputs/train_selected_render_only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path


def _load_case_ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("output_top_k", "ranked_full_pool", "ranked"):
        rows = data.get(key)
        if not rows:
            continue
        out = [str(r["case_id"]) for r in rows]
        return out if key == "output_top_k" else out[:20]
    raise SystemExit(f"无法在 JSON 中找到 output_top_k / ranked_full_pool / ranked: {path}")


def _link_tf_pool(
    repo: Path,
    case_ids: list[str],
    test_pool: list[str],
    seed: int,
) -> None:
    tf_root = repo / "raw/high_quality/medical"
    rng = random.Random(seed)
    for case in case_ids:
        src_case = rng.choice(test_pool)
        src_dir = tf_root / src_case
        src_jsons = sorted(src_dir.glob("*.json"))
        if not src_jsons:
            raise SystemExit(f"TF 源无 json: {src_dir}")
        dst_dir = tf_root / case
        dst_dir.mkdir(parents=True, exist_ok=True)
        for old in dst_dir.glob("*.json"):
            old.unlink()
        for i, src in enumerate(src_jsons, start=1):
            dst = dst_dir / f"{case}_{i}.json"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, required=True, help="select 脚本写出的 JSON 路径")
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Vol2Splat 仓库根（默认脚本上级）",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/train_selected_render_only"),
        help="batch 输出根（相对 cwd 或绝对路径）",
    )
    ap.add_argument(
        "--test-pool",
        type=str,
        default="s0244,s0652,s0693,s0703,s1009,s1073,s1348",
        help="TF 软链源（逗号分隔）",
    )
    ap.add_argument("--link-seed", type=int, default=20260503)
    ap.add_argument("--skip-link-tf", action="store_true", help="不修改 raw/high_quality/medical（你已手动链好 TF 时用）")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="渲染专用 yaml，默认 configs/lqb/high_quality/batch_high_quality_medical_render_only.yaml",
    )
    args = ap.parse_args()

    repo = args.repo_root.resolve()
    json_path = args.json_out if args.json_out.is_absolute() else (repo / args.json_out)
    if not json_path.is_file():
        raise SystemExit(f"找不到 JSON: {json_path}")

    case_ids = _load_case_ids(json_path)
    test_pool = [x.strip() for x in args.test_pool.split(",") if x.strip()]

    if not args.skip_link_tf:
        _link_tf_pool(repo, case_ids, test_pool, seed=args.link_seed)
        print(f"已为 {len(case_ids)} 个训练 case 软链 TF（seed={args.link_seed}）", file=sys.stderr)

    cfg = args.config or (repo / "configs/lqb/high_quality/batch_high_quality_medical_render_only.yaml")
    cfg = cfg if cfg.is_absolute() else (repo / cfg)
    out_root = args.output_root if args.output_root.is_absolute() else (repo / args.output_root)
    case_ids_arg = ",".join(case_ids)

    cmd = [
        sys.executable,
        "-m",
        "vol2splat.cli",
        "batch",
        "--config",
        str(cfg),
        "--raw-root",
        str(repo / "raw/totalseg3d"),
        "--output-root",
        str(out_root),
        "--case-ids",
        case_ids_arg,
        "--no-shuffle",
        "--batch-size",
        str(max(1, len(case_ids))),
    ]
    print("运行:", " ".join(cmd), file=sys.stderr)
    os.chdir(repo)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
