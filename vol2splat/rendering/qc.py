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


def _compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.sqrt(gx * gx + gy * gy)


def _compute_laplacian(gray: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(gray, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return lap


def _compute_fft_high_freq_ratio(gray: np.ndarray, cutoff_ratio: float) -> float:
    if gray.size == 0:
        return 0.0
    fft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(fft) ** 2
    total_power = float(power.sum())
    if total_power <= 1e-12:
        return 0.0
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    cutoff = max(float(min(h, w)) * float(cutoff_ratio), 1.0)
    high_freq_power = float(power[rr >= cutoff].sum())
    return high_freq_power / total_power


def _compute_entropy(gray: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0), density=False)
    total = int(hist.sum())
    if total <= 0:
        return 0.0
    prob = hist.astype(np.float64) / float(total)
    prob = prob[prob > 0.0]
    entropy = float(-(prob * np.log2(prob)).sum())
    return entropy / np.log2(float(bins))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def _masked_std(values: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return 0.0
    return float(values[mask].std())


def _masked_var(values: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not np.any(mask):
        return 0.0
    return float(values[mask].var())


def _compute_image_metrics(
    rgba: np.ndarray,
    pixel_threshold: float,
    alpha_threshold: float,
    fft_cutoff_ratio: float,
) -> Dict[str, float]:
    gray = rgba[..., :3].mean(axis=-1).astype(np.float32, copy=False)
    alpha = rgba[..., 3].astype(np.float32, copy=False)
    foreground_mask = alpha > alpha_threshold

    gradient = _compute_gradient_magnitude(gray)
    laplacian = _compute_laplacian(gray)

    masked_gray = np.where(foreground_mask, gray, 0.0).astype(np.float32, copy=False)
    masked_gradient = _compute_gradient_magnitude(masked_gray)
    masked_laplacian = _compute_laplacian(masked_gray)

    foreground_alpha_mean = _masked_mean(alpha, foreground_mask)
    masked_fft_high_freq_ratio = _compute_fft_high_freq_ratio(masked_gray, fft_cutoff_ratio)
    masked_laplacian_variance = _masked_var(masked_laplacian, foreground_mask)
    detail_over_opacity = (
        masked_laplacian_variance + masked_fft_high_freq_ratio
    ) / max(foreground_alpha_mean, 1e-6)

    return {
        "mean_intensity": float(gray.mean()),
        "nonzero_ratio": float((gray > pixel_threshold).mean()),
        "intensity_std": float(gray.std()),
        "entropy": _compute_entropy(gray),
        "alpha_mean": float(alpha.mean()),
        "alpha_nonzero_ratio": float((alpha > alpha_threshold).mean()),
        "gradient_mean": float(gradient.mean()),
        "laplacian_variance": float(laplacian.var()),
        "fft_high_freq_ratio": _compute_fft_high_freq_ratio(gray, fft_cutoff_ratio),
        "foreground_mean_intensity": _masked_mean(gray, foreground_mask),
        "foreground_intensity_std": _masked_std(gray, foreground_mask),
        "foreground_alpha_mean": foreground_alpha_mean,
        "masked_gradient_mean": _masked_mean(masked_gradient, foreground_mask),
        "masked_laplacian_variance": masked_laplacian_variance,
        "masked_fft_high_freq_ratio": masked_fft_high_freq_ratio,
        "detail_over_opacity": float(detail_over_opacity),
    }


def _apply_metric_rescue_rules(
    metrics: Dict[str, float],
    reasons: List[str],
    qc_cfg: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    remaining = list(reasons)
    rescue_codes: List[str] = []

    if remaining == ["foggy_low_detail"]:
        rescue_alpha_max = qc_cfg.get("rescue_foggy_foreground_alpha_max")
        rescue_min_masked_hf = qc_cfg.get("rescue_foggy_min_masked_fft_high_freq_ratio")
        rescue_min_detail = qc_cfg.get("rescue_foggy_min_detail_over_opacity")
        if (
            rescue_alpha_max is not None
            and rescue_min_masked_hf is not None
            and rescue_min_detail is not None
            and float(metrics.get("foreground_alpha_mean", 0.0)) <= float(rescue_alpha_max)
            and float(metrics.get("masked_fft_high_freq_ratio", 0.0)) >= float(rescue_min_masked_hf)
            and float(metrics.get("detail_over_opacity", 0.0)) >= float(rescue_min_detail)
        ):
            remaining = []
            rescue_codes.append("rescued_foggy_good_structure")

    configured_allowed_peak_reasons = qc_cfg.get("rescue_peak_structure_allowed_reasons")
    if configured_allowed_peak_reasons is None:
        allowed_peak_rescue_reasons = {
            "foggy_low_detail",
            "low_masked_high_frequency",
            "too_dark",
            "too_sparse",
        }
    else:
        if isinstance(configured_allowed_peak_reasons, str):
            configured_allowed_peak_reasons = [configured_allowed_peak_reasons]
        allowed_peak_rescue_reasons = {
            str(reason).strip()
            for reason in configured_allowed_peak_reasons
            if str(reason).strip()
        }
    rescue_peak_alpha_max = qc_cfg.get("rescue_peak_structure_foreground_alpha_max")
    rescue_peak_max_masked_hf = qc_cfg.get("rescue_peak_structure_min_max_masked_fft_high_freq_ratio")
    rescue_peak_max_detail = qc_cfg.get("rescue_peak_structure_min_max_detail_over_opacity")
    rescue_peak_min_nonzero_ratio = qc_cfg.get("rescue_peak_structure_min_nonzero_ratio")
    rescue_peak_min_mean_intensity = qc_cfg.get("rescue_peak_structure_min_mean_intensity")
    if (
        remaining
        and set(remaining).issubset(allowed_peak_rescue_reasons)
        and rescue_peak_alpha_max is not None
        and rescue_peak_max_masked_hf is not None
        and rescue_peak_max_detail is not None
        and rescue_peak_min_nonzero_ratio is not None
        and rescue_peak_min_mean_intensity is not None
        and float(metrics.get("foreground_alpha_mean", 0.0)) <= float(rescue_peak_alpha_max)
        and float(metrics.get("nonzero_ratio", 0.0)) >= float(rescue_peak_min_nonzero_ratio)
        and float(metrics.get("mean_intensity", 0.0)) >= float(rescue_peak_min_mean_intensity)
        and float(metrics.get("max_masked_fft_high_freq_ratio", 0.0)) >= float(rescue_peak_max_masked_hf)
        and float(metrics.get("max_detail_over_opacity", 0.0)) >= float(rescue_peak_max_detail)
    ):
        remaining = []
        rescue_codes.append("rescued_peak_structure_soft_foreground")

    return remaining, rescue_codes


def _analyze_tf_output(tf_output: Dict[str, Any], qc_cfg: Dict[str, Any], segmentation_evaluator=None) -> Dict[str, Any]:
    pixel_threshold = float(qc_cfg.get("pixel_threshold", 0.02))
    min_mean_intensity = float(qc_cfg.get("min_mean_intensity", 0.01))
    min_nonzero_ratio = float(qc_cfg.get("min_nonzero_ratio", 0.01))
    min_intensity_std = float(qc_cfg.get("min_intensity_std", 0.005))
    sample_limit = int(qc_cfg.get("sample_limit", 12))
    alpha_threshold = float(qc_cfg.get("alpha_threshold", 0.02))
    fft_cutoff_ratio = float(qc_cfg.get("fft_cutoff_ratio", 0.12))
    min_gradient_mean = qc_cfg.get("min_gradient_mean")
    min_laplacian_variance = qc_cfg.get("min_laplacian_variance")
    min_fft_high_freq_ratio = qc_cfg.get("min_fft_high_freq_ratio")
    min_masked_laplacian_variance = qc_cfg.get("min_masked_laplacian_variance")
    min_masked_fft_high_freq_ratio = qc_cfg.get("min_masked_fft_high_freq_ratio")
    min_detail_over_opacity = qc_cfg.get("min_detail_over_opacity")

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
                "gradient_mean": 0.0,
                "laplacian_variance": 0.0,
                "fft_high_freq_ratio": 0.0,
                "masked_laplacian_variance": 0.0,
                "masked_fft_high_freq_ratio": 0.0,
                "detail_over_opacity": 0.0,
            },
        }

    aggregated_metrics: Dict[str, List[float]] = {}
    rgba_images = []
    for path in image_paths:
        rgba = _load_image_rgba(path)
        rgba_images.append(rgba)
        image_metrics = _compute_image_metrics(
            rgba,
            pixel_threshold=pixel_threshold,
            alpha_threshold=alpha_threshold,
            fft_cutoff_ratio=fft_cutoff_ratio,
        )
        for key, value in image_metrics.items():
            aggregated_metrics.setdefault(key, []).append(float(value))

    metrics = {key: float(np.mean(values)) for key, values in aggregated_metrics.items()}
    for key in ("masked_fft_high_freq_ratio", "detail_over_opacity"):
        values = aggregated_metrics.get(key)
        if values:
            metrics[f"max_{key}"] = float(np.max(values))
    metrics["num_images"] = len(image_paths)

    mean_intensity = float(metrics["mean_intensity"])
    nonzero_ratio = float(metrics["nonzero_ratio"])
    intensity_std = float(metrics["intensity_std"])

    if nonzero_ratio < min_nonzero_ratio:
        reasons.append("too_sparse")
    if mean_intensity < min_mean_intensity:
        reasons.append("too_dark")
    if intensity_std < min_intensity_std:
        reasons.append("low_contrast")
    if min_gradient_mean is not None and float(metrics["gradient_mean"]) < float(min_gradient_mean):
        reasons.append("low_gradient")
    if min_laplacian_variance is not None and float(metrics["laplacian_variance"]) < float(min_laplacian_variance):
        reasons.append("low_laplacian")
    if min_fft_high_freq_ratio is not None and float(metrics["fft_high_freq_ratio"]) < float(min_fft_high_freq_ratio):
        reasons.append("low_high_frequency")
    if min_masked_laplacian_variance is not None and float(metrics["masked_laplacian_variance"]) < float(min_masked_laplacian_variance):
        reasons.append("low_masked_laplacian")
    if min_masked_fft_high_freq_ratio is not None and float(metrics["masked_fft_high_freq_ratio"]) < float(min_masked_fft_high_freq_ratio):
        reasons.append("low_masked_high_frequency")
    if min_detail_over_opacity is not None and float(metrics["detail_over_opacity"]) < float(min_detail_over_opacity):
        reasons.append("foggy_low_detail")

    segmentation_metrics = None
    if segmentation_evaluator is not None:
        segmentation_metrics = segmentation_evaluator.evaluate(rgba_images)
        if not segmentation_metrics.get("passed", False):
            reasons.append("segmentation_failed")

    reasons, rescue_codes = _apply_metric_rescue_rules(metrics, reasons, qc_cfg)

    return {
        "tf_name": tf_output["tf_name"],
        "tf_json": tf_output["tf_json"],
        "tf_dir": tf_output["tf_dir"],
        "status": "pass" if not reasons else "fail",
        "reason_codes": reasons,
        "rescue_codes": rescue_codes,
        "metrics": {**metrics, "segmentation": segmentation_metrics},
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
        "| TF | Status | Mean | Nonzero | Std | Masked HF | Detail/Opaque | Seg | Rescue | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in report["items"]:
        metrics = item["metrics"]
        reasons = ",".join(item["reason_codes"]) if item["reason_codes"] else "-"
        rescues = ",".join(item.get("rescue_codes") or []) if item.get("rescue_codes") else "-"
        seg_metrics = metrics.get("segmentation")
        if seg_metrics:
            seg_text = f"{seg_metrics['detected_frames']}/{seg_metrics['total_frames']}"
        else:
            seg_text = "-"
        lines.append(
            f"| {item['tf_name']} | {item['status']} | "
            f"{metrics['mean_intensity']:.4f} | {metrics['nonzero_ratio']:.4f} | "
            f"{metrics['intensity_std']:.4f} | {metrics.get('masked_fft_high_freq_ratio', 0.0):.4f} | "
            f"{metrics.get('detail_over_opacity', 0.0):.4f} | {seg_text} | {rescues} | {reasons} |"
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
