import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class SceneImageCheck:
    dataset: str
    scene: str
    checked_images: int
    partial_alpha_pixels: int
    partial_pixels_with_rgb_gt_a: int
    straight_alpha_ratio: float
    status: str
    reason: str
    sample_image: str | None = None
    sample_rgba: list[int] | None = None
    sample_yx: list[int] | None = None


def _iter_dataset_dirs(root: Path, prefix: str):
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            yield path


def _iter_scene_dirs(dataset_dir: Path):
    for tf_config_path in sorted(dataset_dir.glob("**/tf_config.json")):
        scene_dir = tf_config_path.parent
        if scene_dir.is_dir():
            yield scene_dir


def _collect_image_paths(scene_dir: Path, max_images_per_split: int):
    paths = []
    for split in ("train", "test", "val"):
        split_paths = sorted((scene_dir / split).glob("*.png"))
        if max_images_per_split > 0:
            split_paths = split_paths[:max_images_per_split]
        paths.extend(split_paths)
    return paths


def _inspect_scene(scene_dir: Path, max_images_per_split: int) -> SceneImageCheck:
    image_paths = _collect_image_paths(scene_dir, max_images_per_split=max_images_per_split)
    if not image_paths:
        return SceneImageCheck(
            dataset=scene_dir.parent.name,
            scene=scene_dir.name,
            checked_images=0,
            partial_alpha_pixels=0,
            partial_pixels_with_rgb_gt_a=0,
            straight_alpha_ratio=0.0,
            status="fail",
            reason="no_png_images",
        )

    partial_alpha_pixels = 0
    partial_pixels_with_rgb_gt_a = 0
    sample_image = None
    sample_rgba = None
    sample_yx = None

    for image_path in image_paths:
        rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.uint8)
        rgb = rgba[..., :3].astype(np.int32)
        alpha = rgba[..., 3].astype(np.int32)
        partial_mask = (alpha > 0) & (alpha < 255)
        evidence_mask = partial_mask & (rgb > alpha[..., None]).any(axis=-1)

        partial_alpha_pixels += int(partial_mask.sum())
        partial_pixels_with_rgb_gt_a += int(evidence_mask.sum())

        if sample_image is None:
            coords = np.argwhere(evidence_mask)
            if coords.size:
                y, x = coords[0]
                sample_image = image_path.name
                sample_rgba = rgba[y, x].tolist()
                sample_yx = [int(y), int(x)]

    if partial_alpha_pixels == 0:
        return SceneImageCheck(
            dataset=scene_dir.parent.name,
            scene=scene_dir.name,
            checked_images=len(image_paths),
            partial_alpha_pixels=0,
            partial_pixels_with_rgb_gt_a=0,
            straight_alpha_ratio=0.0,
            status="warn",
            reason="no_partial_alpha_pixels",
        )

    ratio = partial_pixels_with_rgb_gt_a / partial_alpha_pixels
    if partial_pixels_with_rgb_gt_a == 0:
        status = "fail"
        reason = "all_checked_partial_pixels_satisfy_rgb_le_alpha"
    else:
        status = "pass"
        reason = "straight_alpha_evidence_found"

    return SceneImageCheck(
        dataset=scene_dir.parent.name,
        scene=scene_dir.name,
        checked_images=len(image_paths),
        partial_alpha_pixels=partial_alpha_pixels,
        partial_pixels_with_rgb_gt_a=partial_pixels_with_rgb_gt_a,
        straight_alpha_ratio=ratio,
        status=status,
        reason=reason,
        sample_image=sample_image,
        sample_rgba=sample_rgba,
        sample_yx=sample_yx,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check that high_quality-prefix output images show straight-alpha evidence instead of premultiplied-alpha behavior."
    )
    parser.add_argument(
        "--root",
        default="outputs",
        help="Root directory that contains high_quality* output datasets.",
    )
    parser.add_argument(
        "--prefix",
        default="high_quality",
        help="Dataset directory prefix to inspect.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=0,
        help="If > 0, only inspect the first N PNGs from each split.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text summary.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    dataset_dirs = list(_iter_dataset_dirs(root, prefix=args.prefix))
    if not dataset_dirs:
        raise SystemExit(f"No dataset directories found under {root} with prefix {args.prefix!r}")

    results = []
    for dataset_dir in dataset_dirs:
        for scene_dir in _iter_scene_dirs(dataset_dir):
            results.append(_inspect_scene(scene_dir, max_images_per_split=args.max_images_per_split))

    if not results:
        raise SystemExit(f"No scene directories with tf_config.json found under datasets matching {args.prefix!r} in {root}")

    if args.json:
        print(json.dumps([asdict(item) for item in results], indent=2, ensure_ascii=False))
    else:
        print(f"Checked {len(results)} scenes under {root} with prefix {args.prefix!r}")
        for item in results:
            line = (
                f"[{item.status.upper()}] {item.dataset}/{item.scene}: "
                f"images={item.checked_images}, "
                f"partial_alpha={item.partial_alpha_pixels}, "
                f"rgb_gt_a={item.partial_pixels_with_rgb_gt_a}, "
                f"ratio={item.straight_alpha_ratio:.4f}, "
                f"reason={item.reason}"
            )
            print(line)
            if item.sample_image is not None:
                print(
                    f"  sample={item.sample_image} rgba={item.sample_rgba} yx={item.sample_yx}"
                )

    has_failure = any(item.status == "fail" for item in results)
    if has_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
