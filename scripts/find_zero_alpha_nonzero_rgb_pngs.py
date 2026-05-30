#!/usr/bin/env python3
"""
定位 PNG 中 `alpha == 0` 但 `rgb != 0` 的像素。

默认递归扫描给定目录下的所有 PNG，并输出：
  - 命中的 PNG 路径
  - 每张图命中的像素数量
  - 若干示例像素的 `(y, x)` 与 `RGBA`

示例：
  python scripts/find_zero_alpha_nonzero_rgb_pngs.py \
    --root outputs/high_quality_sim_vti_cache_rule2

  python scripts/find_zero_alpha_nonzero_rgb_pngs.py \
    --root outputs \
    --path-glob '**/test/*.png' \
    --json-out outputs/a0_rgb_nonzero_report.

结果： 定位到了PNG中有RGB非零的像素，但 alpha == 0。 所以数据集中的PNG为C_straight。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _iter_png_paths(root: Path, path_glob: str | None) -> list[Path]:
    if path_glob:
        return sorted(path for path in root.glob(path_glob) if path.is_file())
    return sorted(path for path in root.glob("**/*.png") if path.is_file())


def _inspect_png(path: Path, max_examples_per_image: int) -> dict[str, Any] | None:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)
    rgb = rgba[..., :3].astype(np.int32)
    alpha = rgba[..., 3].astype(np.int32)
    mask = (alpha == 0) & (rgb.sum(axis=-1) > 0)
    hit_count = int(mask.sum())
    if hit_count == 0:
        return None

    examples = []
    coords = np.argwhere(mask)
    for y, x in coords[:max_examples_per_image]:
        examples.append(
            {
                "yx": [int(y), int(x)],
                "rgba": rgba[y, x].tolist(),
            }
        )

    rgb_hit = rgb[mask]
    return {
        "path": str(path),
        "hit_pixels": hit_count,
        "rgb_min_on_hits": rgb_hit.min(axis=0).tolist(),
        "rgb_max_on_hits": rgb_hit.max(axis=0).tolist(),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find PNGs containing pixels where alpha==0 but RGB is nonzero.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory to scan recursively.",
    )
    parser.add_argument(
        "--path-glob",
        type=str,
        default=None,
        help="Optional glob relative to --root, for example '**/test/*.png'.",
    )
    parser.add_argument(
        "--max-examples-per-image",
        type=int,
        default=5,
        help="Number of sample hit pixels to keep for each PNG.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="If > 0, only print the first N hit PNGs in text mode. JSON still contains all hits.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the full report as JSON.",
    )
    parser.add_argument(
        "--fail-if-found",
        action="store_true",
        help="Exit with code 1 if any hit is found.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    png_paths = _iter_png_paths(root, args.path_glob)
    results: list[dict[str, Any]] = []
    total_hit_pixels = 0
    for path in png_paths:
        item = _inspect_png(path, max_examples_per_image=max(1, args.max_examples_per_image))
        if item is None:
            continue
        results.append(item)
        total_hit_pixels += int(item["hit_pixels"])

    report = {
        "root": str(root),
        "path_glob": args.path_glob,
        "png_scanned": len(png_paths),
        "images_with_hits": len(results),
        "total_hit_pixels": total_hit_pixels,
        "items": results,
    }

    if args.json_out is not None:
        json_path = args.json_out.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {json_path}", file=sys.stderr)

    print(f"scanned_png={len(png_paths)}")
    print(f"images_with_a0_rgb_nonzero={len(results)}")
    print(f"pixels_with_a0_rgb_nonzero={total_hit_pixels}")

    shown = results if args.max_images <= 0 else results[: args.max_images]
    for item in shown:
        print(f"\n{item['path']}")
        print(f"  hit_pixels={item['hit_pixels']}")
        print(f"  rgb_min_on_hits={item['rgb_min_on_hits']}")
        print(f"  rgb_max_on_hits={item['rgb_max_on_hits']}")
        for ex in item["examples"]:
            print(f"  sample yx={ex['yx']} rgba={ex['rgba']}")

    if args.fail_if_found and results:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
