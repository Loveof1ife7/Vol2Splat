from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional
import numpy as np
try:
    import torch
except Exception:
    torch = None

def has_torch_tf() -> bool:
    return torch is not None

def load_fused_tf_json(tf_json_path: str) -> Dict[str, Any]:
    tf_json_path = os.path.abspath(tf_json_path)
    with open(tf_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    control_points = np.asarray(config.get('control_points', []), dtype=np.float32)
    if control_points.ndim != 2 or control_points.shape[1] != 5:
        raise ValueError(f'Expected fused TF control_points with shape (N, 5), got {control_points.shape}')
    order = np.argsort(control_points[:, 0], kind='stable')
    control_points = control_points[order]
    data_range = config.get('data_range') or [float(control_points[0, 0]), float(control_points[-1, 0])]
    return {'path': tf_json_path, 'data_range': [float(data_range[0]), float(data_range[1])], 'control_points': control_points}

def apply_fused_tf_numpy(values: np.ndarray, control_points: np.ndarray, premultiply_alpha: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    control_points = np.asarray(control_points, dtype=np.float32)
    x_coords = control_points[:, 0]
    rgba_cp = control_points[:, 1:5]
    flat_values = values.reshape(-1)
    rgba_flat = np.empty((flat_values.shape[0], 4), dtype=np.float32)
    for c in range(4):
        rgba_flat[:, c] = np.interp(flat_values, x_coords, rgba_cp[:, c], left=float(rgba_cp[0, c]), right=float(rgba_cp[-1, c]))
    if premultiply_alpha:
        rgba_flat[:, :3] *= rgba_flat[:, 3:4]
    return rgba_flat.reshape(values.shape + (4,))

def bake_rgba_volume_numpy(scalar_volume_zyx: np.ndarray, tf_config: Dict[str, Any], premultiply_alpha: bool = False) -> np.ndarray:
    rgba_zyx4 = apply_fused_tf_numpy(scalar_volume_zyx, tf_config['control_points'], premultiply_alpha=premultiply_alpha)
    return np.moveaxis(rgba_zyx4, -1, 0).astype(np.float32, copy=False)

def _ensure_torch_available():
    if torch is None:
        raise ImportError('torch-based transfer function acceleration is unavailable')

def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return str(device)
    if torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def _tf_config_from_input(tf_json_path: Optional[str] = None, tf_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if tf_config is not None:
        return {'path': tf_config.get('path'), 'data_range': list(tf_config['data_range']), 'control_points': np.asarray(tf_config['control_points'], dtype=np.float32)}
    if tf_json_path is None:
        raise ValueError('Expected either tf_json_path or tf_config')
    return load_fused_tf_json(tf_json_path)

def make_json_tf(json_path: Optional[str] = None, tf_config: Optional[Dict[str, Any]] = None):
    _ensure_torch_available()
    resolved = _tf_config_from_input(tf_json_path=json_path, tf_config=tf_config)
    x_coords = torch.tensor(resolved['control_points'][:, 0], dtype=torch.float32)
    rgba_values = torch.tensor(resolved['control_points'][:, 1:5], dtype=torch.float32)
    def tf(s: 'torch.Tensor'):
        x = x_coords.to(device=s.device, dtype=torch.float32)
        rgba = rgba_values.to(device=s.device, dtype=torch.float32)
        right_indices = torch.searchsorted(x, s)
        left_indices = (right_indices - 1).clamp(min=0, max=len(x) - 1)
        right_indices = right_indices.clamp(min=0, max=len(x) - 1)
        x_left, rgba_left = x[left_indices], rgba[left_indices]
        x_right, rgba_right = x[right_indices], rgba[right_indices]
        t = ((s - x_left) / (x_right - x_left + 1e-9)).clamp(0, 1).unsqueeze(-1)
        return rgba_left + t * (rgba_right - rgba_left)
    return tf

def make_gaussian_tf(center: float, width: float, color=(0.2, 0.8, 1.0), opacity_scale: float = 1.0):
    _ensure_torch_available()
    c = torch.tensor(color, dtype=torch.float32)
    mu = torch.tensor(center, dtype=torch.float32)
    sigma = torch.tensor(width, dtype=torch.float32)
    k = torch.tensor(opacity_scale, dtype=torch.float32)
    def tf(s: 'torch.Tensor'):
        c_, mu_, sigma_, k_ = c.to(s.device, s.dtype), mu.to(s.device, s.dtype), sigma.to(s.device, s.dtype), k.to(s.device, s.dtype)
        w = torch.exp(-0.5 * ((s - mu_) / sigma_) ** 2)
        rgb = c_ * w.unsqueeze(-1)
        opacity = 1.0 - torch.exp(-k_ * w)
        return torch.cat([rgb, opacity.unsqueeze(-1)], dim=-1)
    return tf

def apply_tf_volume(vol: 'torch.Tensor', tf_callable, premultiply_alpha: bool = True, chunk_size: int = int(1e8), block_size: int = 16):
    _ensure_torch_available()
    d_dim, h_dim, w_dim = vol.shape
    total_elements = d_dim * h_dim * w_dim
    if vol.device.type == 'cuda':
        total_gpu_memory = torch.cuda.get_device_properties(vol.device).total_memory
        available_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated(vol.device)
        estimated_memory = total_elements * 20
        mode = 'block' if estimated_memory > available_gpu_memory * 0.7 else 'chunk'
    else:
        mode = 'chunk'
    if mode == 'block':
        out = torch.empty((d_dim, h_dim, w_dim, 4), device=vol.device, dtype=torch.float32)
        for d0 in range(0, d_dim, block_size):
            d1 = min(d_dim, d0 + block_size)
            rgba = tf_callable(vol[d0:d1].reshape(-1))
            if premultiply_alpha:
                rgba[:, :3] *= rgba[:, 3:4]
            out[d0:d1] = rgba.reshape(d1 - d0, h_dim, w_dim, 4)
        return out
    rgba_chunks = []
    for chunk in torch.split(vol.reshape(-1), chunk_size):
        rgba = tf_callable(chunk)
        if premultiply_alpha:
            rgba[:, :3] *= rgba[:, 3:4]
        rgba_chunks.append(rgba)
    return torch.cat(rgba_chunks, dim=0).reshape(d_dim, h_dim, w_dim, 4)

def apply_fused_tf_torch(values: 'torch.Tensor', control_points: np.ndarray | 'torch.Tensor', premultiply_alpha: bool = False) -> 'torch.Tensor':
    _ensure_torch_available()
    cp = control_points if isinstance(control_points, torch.Tensor) else torch.as_tensor(control_points, dtype=torch.float32, device=values.device)
    x_coords = cp[:, 0]
    rgba_cp = cp[:, 1:5]
    flat_values = values.reshape(-1).to(dtype=torch.float32)
    right_indices = torch.searchsorted(x_coords, flat_values)
    left_indices = (right_indices - 1).clamp(min=0, max=x_coords.shape[0] - 1)
    right_indices = right_indices.clamp(min=0, max=x_coords.shape[0] - 1)
    x_left = x_coords[left_indices]
    x_right = x_coords[right_indices]
    rgba_left = rgba_cp[left_indices]
    rgba_right = rgba_cp[right_indices]
    t = ((flat_values - x_left) / (x_right - x_left + 1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    rgba_flat = rgba_left + t * (rgba_right - rgba_left)
    if premultiply_alpha:
        rgba_flat[:, :3] *= rgba_flat[:, 3:4]
    return rgba_flat.reshape(values.shape + (4,))

def bake_rgba_volume_torch(scalar_volume_zyx: np.ndarray | 'torch.Tensor', tf_json_path: Optional[str] = None, tf_config: Optional[Dict[str, Any]] = None, premultiply_alpha: bool = False, device: Optional[str] = None, return_numpy: bool = True):
    _ensure_torch_available()
    resolved = _tf_config_from_input(tf_json_path=tf_json_path, tf_config=tf_config)
    values = scalar_volume_zyx if isinstance(scalar_volume_zyx, torch.Tensor) else torch.as_tensor(scalar_volume_zyx, dtype=torch.float32)
    values = values.to(device=_resolve_device(device), dtype=torch.float32)
    rgba_zyx4 = apply_tf_volume(values, make_json_tf(tf_config=resolved), premultiply_alpha=premultiply_alpha)
    rgba_4zyx = rgba_zyx4.permute(3, 0, 1, 2).contiguous()
    return rgba_4zyx.detach().cpu().numpy() if return_numpy else rgba_4zyx

def bake_rgba_volume(scalar_volume_zyx: np.ndarray | 'torch.Tensor', tf_json_path: Optional[str] = None, tf_config: Optional[Dict[str, Any]] = None, premultiply_alpha: bool = False, backend: str = 'auto', device: Optional[str] = None, return_numpy: bool = True):
    if backend not in {'auto', 'torch', 'numpy'}:
        raise ValueError(f'Unsupported TF backend: {backend}')
    if backend == 'torch' or (backend == 'auto' and torch is not None):
        return bake_rgba_volume_torch(scalar_volume_zyx, tf_json_path=tf_json_path, tf_config=tf_config, premultiply_alpha=premultiply_alpha, device=device, return_numpy=return_numpy)
    resolved = _tf_config_from_input(tf_json_path=tf_json_path, tf_config=tf_config)
    values = scalar_volume_zyx if isinstance(scalar_volume_zyx, np.ndarray) else scalar_volume_zyx.detach().cpu().numpy()
    return bake_rgba_volume_numpy(values, resolved, premultiply_alpha=premultiply_alpha)
