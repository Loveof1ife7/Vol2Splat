from typing import Any, Dict, Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from scipy.ndimage import zoom as scipy_zoom
except Exception:
    scipy_zoom = None

from ..core.pipeline import Stage
from ..core.types import Volume
from ..export.write_vti import write_volume_to_vti
from ..registry import register_stage


def _parse_spacing(value: Any, default: Sequence[float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    if isinstance(value, (int, float)):
        out = np.array([float(value), float(value), float(value)], dtype=np.float64)
    else:
        if len(value) != 3:
            raise ValueError(f"target_spacing must be scalar or length-3, got {value}")
        out = np.asarray(value, dtype=np.float64)
    if np.any(out <= 0):
        raise ValueError(f"target_spacing must be positive, got {out}")
    return out


class CanonicalizeStage(Stage):
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        original_shape = tuple(int(v) for v in vol.data.shape)
        original_spacing = np.asarray(vol.spacing, dtype=np.float64)
        target_spacing = _parse_spacing(cfg.get("target_spacing", 1.0), default=original_spacing)
        backend = str(cfg.get("backend", "auto")).lower()
        mode = str(cfg.get("mode", "trilinear")).lower()
        align_corners = bool(cfg.get("align_corners", True))
        keep_dtype = bool(cfg.get("keep_dtype", False))
        pad_to_cube = bool(cfg.get("pad_to_cube", False))
        cube_size = cfg.get("cube_size")
        write_vti = bool(cfg.get("write_vti", False))
        vti_path = cfg.get("vti_path")

        resampled = self._resample(vol.data, original_spacing, target_spacing, backend, mode, align_corners)
        if pad_to_cube or cube_size:
            resampled = self._pad_to_cube(resampled, cube_size)

        if keep_dtype:
            resampled = resampled.astype(vol.data.dtype, copy=False)
        else:
            resampled = resampled.astype(np.float32, copy=False)

        vol.data = resampled
        vol.spacing = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        vol.origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        vol.direction = np.eye(3, dtype=np.float64)
        vol.cache["canonicalized"] = True
        vol.metadata.extra["canonical_space"] = {
            "enabled": True,
            "original_shape": list(original_shape),
            "original_spacing_xyz": original_spacing.tolist(),
            "target_spacing_xyz": target_spacing.tolist(),
            "output_shape": list(vol.data.shape),
            "spacing_xyz": [1.0, 1.0, 1.0],
            "origin_xyz": [0.0, 0.0, 0.0],
            "direction": np.eye(3, dtype=np.float64).tolist(),
        }

        if write_vti or vti_path:
            if not vti_path:
                raise ValueError("canonicalize requires 'vti_path' when write_vti is enabled")
            write_volume_to_vti(vol, vti_path)
            vol.cache["canonical_vti_path"] = vti_path
            vol.metadata.extra["canonical_vti_path"] = vti_path

        return vol

    def _resample(self, data, original_spacing, target_spacing, backend, mode, align_corners):
        '''
        Keep physical extent approximately consistent:
        new_size_xyz ≈ old_size_xyz * old_spacing_xyz / target_spacing_xyz
        zoom_xyz = old_spacing_xyz / target_spacing_xyz
        ndarray is (z, y, x), so convert xyz -> zyx for interpolation.
        '''
        zoom_xyz = original_spacing / target_spacing
        target_shape_zyx = np.maximum(
            1,
            np.round(np.array(data.shape[-3:], dtype=np.float64) * np.array([zoom_xyz[2], zoom_xyz[1], zoom_xyz[0]], dtype=np.float64)).astype(int),
        )
        if backend == "auto":
            backend = "torch" if torch is not None else "scipy"
        if backend == "torch":
            if torch is None or F is None:
                raise ImportError("torch backend requested for canonicalize but torch is unavailable")
            return self._resample_torch(data, tuple(int(v) for v in target_shape_zyx), mode, align_corners)
        if backend == "scipy":
            if scipy_zoom is None:
                raise ImportError("scipy backend requested for canonicalize but scipy is unavailable")
            return self._resample_scipy(data, tuple(int(v) for v in target_shape_zyx), mode)
        raise ValueError(f"Unsupported canonicalize backend: {backend}")

    def _resample_torch(self, data, target_shape_zyx, mode, align_corners):
        tensor = torch.as_tensor(data, dtype=torch.float32)
        interpolation_mode = "nearest" if mode == "nearest" else "trilinear"
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            out = F.interpolate(tensor, size=target_shape_zyx, mode=interpolation_mode, align_corners=align_corners if interpolation_mode != "nearest" else None)
            return out.squeeze(0).squeeze(0).cpu().numpy()
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
            out = F.interpolate(tensor, size=target_shape_zyx, mode=interpolation_mode, align_corners=align_corners if interpolation_mode != "nearest" else None)
            return out.squeeze(0).cpu().numpy()
        raise ValueError(f"canonicalize expects 3D or 4D volume, got {data.shape}")

    def _resample_scipy(self, data, target_shape_zyx, mode):
        zoom_zyx = np.asarray(target_shape_zyx, dtype=np.float64) / np.asarray(data.shape[-3:], dtype=np.float64)
        order = 0 if mode == "nearest" else 1
        if data.ndim == 3:
            return scipy_zoom(data.astype(np.float32, copy=False), zoom=zoom_zyx, order=order)
        if data.ndim == 4:
            return scipy_zoom(data.astype(np.float32, copy=False), zoom=(1.0, zoom_zyx[0], zoom_zyx[1], zoom_zyx[2]), order=order)
        raise ValueError(f"canonicalize expects 3D or 4D volume, got {data.shape}")

    def _pad_to_cube(self, data, cube_size):
        z_dim, y_dim, x_dim = data.shape[-3:]
        target = int(cube_size) if cube_size else int(max(z_dim, y_dim, x_dim))
        if target < max(z_dim, y_dim, x_dim):
            raise ValueError(f"cube_size={target} is smaller than current volume shape {(z_dim, y_dim, x_dim)}")
        pads = []
        for size in [x_dim, y_dim, z_dim]:
            total = target - size
            left = total // 2
            right = total - left
            pads.append((left, right))
        pad_zyx = (pads[2], pads[1], pads[0])
        if data.ndim == 3:
            return np.pad(data, pad_zyx, mode="constant", constant_values=0)
        return np.pad(data, ((0, 0),) + pad_zyx, mode="constant", constant_values=0)


register_stage("canonicalize", CanonicalizeStage)
