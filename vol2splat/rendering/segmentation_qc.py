from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from monai.networks.nets import UNet
except Exception:
    UNet = None


def _require_runtime() -> None:
    if torch is None:
        raise ImportError("torch is required for segmentation-aware render QC")
    if Image is None:
        raise ImportError("Pillow is required for segmentation-aware render QC")
    if UNet is None:
        raise ImportError("monai is required for MONAI-UNet segmentation-aware render QC")


def _resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resize_image_hwc(image_hwc: np.ndarray, input_size) -> np.ndarray:
    if input_size is None:
        return image_hwc.astype(np.float32, copy=False)
    if len(input_size) != 2:
        raise ValueError(f"segmentation.input_size must be [H, W], got {input_size}")
    height, width = int(input_size[0]), int(input_size[1])
    image_uint8 = np.clip(image_hwc * 255.0, 0.0, 255.0).astype(np.uint8)
    if image_uint8.ndim == 2:
        pil = Image.fromarray(image_uint8, mode="L")
        resized = np.asarray(pil.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
        return resized

    mode = "RGBA" if image_uint8.shape[2] == 4 else "RGB"
    pil = Image.fromarray(image_uint8, mode=mode)
    resized = np.asarray(pil.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    return resized


def _prepare_tensor(image_hwc: np.ndarray, cfg: Dict[str, Any], device: str):
    resized = _resize_image_hwc(np.asarray(image_hwc, dtype=np.float32), cfg.get("input_size"))
    if resized.ndim == 2:
        resized = resized[:, :, None]

    in_channels = int(cfg.get("in_channels", 4))
    rgb = resized[..., :3]
    alpha = resized[..., 3:4] if resized.shape[2] >= 4 else np.ones((*resized.shape[:2], 1), dtype=np.float32)
    gray = rgb.mean(axis=-1, keepdims=True)

    if in_channels == 1:
        chw = np.transpose(gray, (2, 0, 1))
    elif in_channels == 3:
        chw = np.transpose(rgb, (2, 0, 1))
    elif in_channels == 4:
        rgba = np.concatenate([rgb, alpha], axis=-1)
        chw = np.transpose(rgba, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported MONAI segmentation in_channels: {in_channels}")

    tensor = torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)
    mean = cfg.get("mean")
    std = cfg.get("std")
    if mean is not None and std is not None:
        mean_tensor = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor.clamp_min(1e-6)
    return tensor


def _unwrap_output(output):
    if isinstance(output, dict):
        for key in ("logits", "pred", "out", "mask"):
            if key in output:
                return output[key]
        return next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, OrderedDict):
        return checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported MONAI segmentation checkpoint format")


def _strip_common_prefixes(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    stripped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "net."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        stripped[new_key] = value
    return stripped


class MonaiUNetSegmentationEvaluator:
    def __init__(self, cfg: Dict[str, Any]):
        _require_runtime()
        model_path = cfg.get("model_path") or cfg.get("checkpoint_path")
        if not model_path:
            raise ValueError("MONAI segmentation-aware QC requires 'model_path' or 'checkpoint_path'")

        self.cfg = dict(cfg)
        self.device = _resolve_device(cfg.get("device"))

        channels = tuple(int(v) for v in cfg.get("channels", (32, 64, 128, 256)))
        strides = tuple(int(v) for v in cfg.get("strides", (2, 2, 2)))
        if len(channels) != len(strides) + 1:
            raise ValueError("MONAI UNet expects len(channels) == len(strides) + 1")

        self.model = UNet(
            spatial_dims=2,
            in_channels=int(cfg.get("in_channels", 4)),
            out_channels=int(cfg.get("out_channels", 2)),
            channels=channels,
            strides=strides,
            num_res_units=int(cfg.get("num_res_units", 0)),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = _strip_common_prefixes(_extract_state_dict(checkpoint))
        self.model.load_state_dict(state_dict, strict=bool(cfg.get("strict", True)))
        self.model.eval()

    def _infer_single(self, image_hwc: np.ndarray) -> Dict[str, float]:
        tensor = _prepare_tensor(image_hwc, self.cfg, self.device)
        with torch.no_grad():
            output = _unwrap_output(self.model(tensor))
        if output.ndim == 3:
            output = output.unsqueeze(1)
        if output.ndim != 4:
            raise ValueError(f"Unexpected MONAI segmentation output shape: {tuple(output.shape)}")

        mask_threshold = float(self.cfg.get("mask_threshold", 0.5))
        min_confidence = float(self.cfg.get("min_confidence", 0.5))
        min_foreground_ratio = float(self.cfg.get("min_foreground_ratio", 0.001))

        if output.shape[1] == 1:
            prob = torch.sigmoid(output[0, 0])
        else:
            target_class = int(self.cfg.get("target_class", 1))
            probs = torch.softmax(output[0], dim=0)
            if target_class < 0 or target_class >= probs.shape[0]:
                raise ValueError(f"target_class {target_class} out of range for segmentation output with {probs.shape[0]} classes")
            prob = probs[target_class]

        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        mask = prob_np >= mask_threshold
        foreground_ratio = float(mask.mean())
        confidence = float(prob_np[mask].mean()) if np.any(mask) else 0.0
        detected = bool(foreground_ratio >= min_foreground_ratio and confidence >= min_confidence)
        return {
            "detected": detected,
            "foreground_ratio": foreground_ratio,
            "confidence": confidence,
        }

    def evaluate(self, rgba_images: List[np.ndarray]) -> Dict[str, Any]:
        min_detected_frames = int(self.cfg.get("min_detected_frames", 1))
        min_detected_ratio = float(self.cfg.get("min_detected_ratio", 0.1))

        per_image = [self._infer_single(image) for image in rgba_images]
        detected_frames = int(sum(1 for item in per_image if item["detected"]))
        total_frames = len(per_image)
        detected_ratio = float(detected_frames / total_frames) if total_frames > 0 else 0.0
        mean_foreground_ratio = float(np.mean([item["foreground_ratio"] for item in per_image])) if per_image else 0.0
        mean_confidence = float(np.mean([item["confidence"] for item in per_image])) if per_image else 0.0
        passed = bool(total_frames > 0 and detected_frames >= min_detected_frames and detected_ratio >= min_detected_ratio)
        return {
            "enabled": True,
            "backend": "monai_unet",
            "passed": passed,
            "detected_frames": detected_frames,
            "total_frames": total_frames,
            "detected_ratio": detected_ratio,
            "mean_foreground_ratio": mean_foreground_ratio,
            "mean_confidence": mean_confidence,
        }


def build_segmentation_evaluator(segmentation_cfg: Dict[str, Any] | None):
    if not segmentation_cfg or not bool(segmentation_cfg.get("enabled", False)):
        return None
    backend = str(segmentation_cfg.get("backend", "monai_unet")).lower()
    if backend not in {"monai", "monai_unet"}:
        raise ValueError(f"Unsupported segmentation QC backend: {backend}")
    return MonaiUNetSegmentationEvaluator(segmentation_cfg)
