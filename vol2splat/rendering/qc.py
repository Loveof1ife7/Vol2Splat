import json
import os
from glob import glob
from typing import Any, Dict, List

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

from .segmentation_qc import build_segmentation_evaluator


def _sample_paths(paths: List[str], limit: int | None) -> List[str]:
    if limit is None or limit <= 0 or len(paths) <= limit:
        return paths
    idx = np.linspace(0, len(paths) - 1, limit).round().astype(int)
    return [paths[int(i)] for i in idx]


def _load_image_gray(path: str) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to run render QC on PNG outputs")
    image = _load_image_rgba(path)
    gray = image[..., :3].astype(np.float32).mean(axis=-1)
    if gray.max() > 1.0:
        gray = gray / 255.0
    return gray


def _load_image_rgba(path: str) -> np.ndarray:
    if Image is None:
        raise ImportError("Pillow is required to run render QC on PNG outputs")
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return array


def _collect_tf_image_paths(tf_dir: str) -> List[str]:
    paths: List[str] = []
    for split in ("train", "test", "val"):
        paths.extend(sorted(glob(os.path.join(tf_dir, split, "*.png"))))
    return paths


def _analyze_tf_output(tf_output: Dict[str, Any], qc_cfg: Dict[str, Any], segmentation_evaluator=None) -> Dict[str, Any]:
    pixel_threshold = float(qc_cfg.get("pixel_threshold", 0.02))
    min_mean_intensity = float(qc_cfg.get("min_mean_intensity", 0.01))
    min_nonzero_ratio = float(qc_cfg.get("min_nonzero_ratio", 0.01))
    min_intensity_std = float(qc_cfg.get("min_intensity_std", 0.005))
    sample_limit = int(qc_cfg.get("sample_limit", 12))

    image_paths = _sample_paths(_collect_tf_image_paths(tf_output["tf_dir"]), sample_limit)
    reasons: List[str] = []
    if not image_paths:
        reasons.append("no_images")
        return {
            "tf_name": tf_output["tf_name"],
            "tf_json": tf_output["tf_json"],
            "tf_dir": tf_output["tf_dir"],
            "status": "fail",
            "reason_codes": reasons,
            "metrics": {
                "num_images": 0,
                "mean_intensity": 0.0,
                "nonzero_ratio": 0.0,
                "intensity_std": 0.0,
            },
        }

    means = []
    nonzero_ratios = []
    stds = []
    rgba_images = []
    for path in image_paths:
        rgba = _load_image_rgba(path)
        gray = rgba[..., :3].mean(axis=-1)
        rgba_images.append(rgba)
        means.append(float(gray.mean()))
        nonzero_ratios.append(float((gray > pixel_threshold).mean()))
        stds.append(float(gray.std()))

    mean_intensity = float(np.mean(means))
    nonzero_ratio = float(np.mean(nonzero_ratios))
    intensity_std = float(np.mean(stds))

    if nonzero_ratio < min_nonzero_ratio:
        reasons.append("too_sparse")
    if mean_intensity < min_mean_intensity:
        reasons.append("too_dark")
    if intensity_std < min_intensity_std:
        reasons.append("low_contrast")

    segmentation_metrics = None
    if segmentation_evaluator is not None:
        segmentation_metrics = segmentation_evaluator.evaluate(rgba_images)
        if not segmentation_metrics.get("passed", False):
            reasons.append("segmentation_failed")

    return {
        "tf_name": tf_output["tf_name"],
        "tf_json": tf_output["tf_json"],
        "tf_dir": tf_output["tf_dir"],
        "status": "pass" if not reasons else "fail",
        "reason_codes": reasons,
        "metrics": {
            "num_images": len(image_paths),
            "mean_intensity": mean_intensity,
            "nonzero_ratio": nonzero_ratio,
            "intensity_std": intensity_std,
            "segmentation": segmentation_metrics,
        },
    }


def _write_qc_markdown(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "# Render QC Report",
        "",
        f"- output_dir: `{report['output_dir']}`",
        f"- total_tf: {report['total_tf']}",
        f"- passed_tf: {report['passed_tf']}",
        f"- failed_tf: {report['failed_tf']}",
        "",
        "| TF | Status | Mean | Nonzero | Std | Seg | Reasons |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report["items"]:
        metrics = item["metrics"]
        reasons = ",".join(item["reason_codes"]) if item["reason_codes"] else "-"
        seg_metrics = metrics.get("segmentation")
        if seg_metrics:
            seg_text = f"{seg_metrics['detected_frames']}/{seg_metrics['total_frames']}"
        else:
            seg_text = "-"
        lines.append(
            f"| {item['tf_name']} | {item['status']} | "
            f"{metrics['mean_intensity']:.4f} | {metrics['nonzero_ratio']:.4f} | "
            f"{metrics['intensity_std']:.4f} | {seg_text} | {reasons} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_render_qc(render_result: Dict[str, Any], qc_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tf_outputs = render_result.get("tf_outputs") or []
    output_dir = os.path.abspath(render_result.get("output_dir") or ".")
    segmentation_evaluator = build_segmentation_evaluator(qc_cfg.get("segmentation"))
    items = [_analyze_tf_output(item, qc_cfg, segmentation_evaluator=segmentation_evaluator) for item in tf_outputs]
    passed_tf_names = [item["tf_name"] for item in items if item["status"] == "pass"]
    failed_tf_names = [item["tf_name"] for item in items if item["status"] != "pass"]

    report = {
        "enabled": True,
        "skip_failed_tf": bool(qc_cfg.get("skip_failed_tf", True)),
        "output_dir": output_dir,
        "total_tf": len(items),
        "passed_tf": len(passed_tf_names),
        "failed_tf": len(failed_tf_names),
        "passed_tf_names": passed_tf_names,
        "failed_tf_names": failed_tf_names,
        "items": items,
    }

    report_json = qc_cfg.get("report_json") or os.path.join(output_dir, "render_qc.json")
    report_md = qc_cfg.get("report_md") or os.path.join(output_dir, "render_qc.md")
    os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_md)), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_qc_markdown(report_md, report)
    report["report_json"] = os.path.abspath(report_json)
    report["report_md"] = os.path.abspath(report_md)
    return report
