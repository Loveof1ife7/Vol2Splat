import json
import torch

@torch.no_grad()
def apply_tf_volume(
    vol : torch.tensor, # [D, H ,W]
    tf_callable,
    premultiply_alpha=False,
    chunk_size=int(1e8),
    block_size=int(16),
):
    assert vol.ndim == 3
    D, H, W = vol.shape
    total_elements = D * H * W
    
    if vol.device.type == "cuda":
        total_gpu_memory = torch.cuda.get_device_properties(vol.device).total_memory
        available_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated(vol.device)
        estimated_memory = total_elements * 20 
        if estimated_memory > available_gpu_memory * 0.7:
            mode = "block"
        else: 
            mode = "chunk"
    else:
        # Fallback for CPU
        mode = "chunk"

    if mode  == 'block':
        out = torch.empty((D, H, W, 4), device=vol.device, dtype=torch.float32)
        for d0 in range(0, D, block_size):
            d1 = min(D, d0 + block_size)
            slab = vol[d0:d1]
            s = slab.reshape(-1)
            rgba = tf_callable(s)
            if premultiply_alpha: rgba[:, :3] *= rgba[:, 3:4]
            out[d0:d1] = rgba
        return out
    
    if mode == 'chunk':
        vol_flat = vol.reshape(-1)
        chunks = torch.split(vol_flat, chunk_size)
        rgba_chunks = []
        for s in chunks:
            rgba = tf_callable(s)
            if premultiply_alpha:
                rgba[:, :3] *= rgba[:, 3:4]
            rgba_chunks.append(rgba)
        rgba_full = torch.cat(rgba_chunks, dim=0).reshape(vol.shape[0], vol.shape[1], vol.shape[2], 4)
        return rgba_full

@torch.no_grad()
def make_json_tf(json_path: str):
    """
    Args:
        json_path: str
    Returns:
        callable: tf(s: torch.Tensor) -> torch.Tensor
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    control_points = torch.tensor(config['control_points'], dtype=torch.float32)
    sorted_indices = torch.argsort(control_points[:, 0])
    control_points = control_points[sorted_indices]
    x_coords = control_points[:, 0]
    rgba_values = control_points[:, 1:]

    print(f"TransferFunction loaded from {json_path} with {len(x_coords)} control points.")

    def tf(s: torch.Tensor):
        device = s.device
        x = x_coords.to(device)
        rgba = rgba_values.to(device)

        right_indices = torch.searchsorted(x, s)
        left_indices = right_indices - 1

        left_indices = left_indices.clamp(min=0, max=len(x) - 1)
        right_indices = right_indices.clamp(min=0, max=len(x) - 1)

        x_left, rgba_left = x[left_indices], rgba[left_indices]
        x_right, rgba_right = x[right_indices], rgba[right_indices]

        t = (s - x_left) / (x_right - x_left + 1e-9)
        t = t.clamp(0, 1).unsqueeze(-1)

        interpolated_rgba = rgba_left + t * (rgba_right - rgba_left)
        return interpolated_rgba
    return tf

@torch.no_grad()
def make_gaussian_tf(
    center: float,
    width: float,
    color=(0.2, 0.8, 1.0),
    opacity_scale: float = 1.0,
):
    """
    Transfer Function: density → RGBA 
    - input: s (Tensor)
    - output: [r,g,b,opacity]
    - 4th channel =  (opacity), range [0,1]
    """
    c = torch.tensor(color, dtype=torch.float32)
    mu = torch.tensor(center, dtype=torch.float32)
    sigma = torch.tensor(width, dtype=torch.float32)
    k = torch.tensor(opacity_scale, dtype=torch.float32)

    def tf(s: torch.Tensor):
        # broadcast to match device/dtype
        device, dtype = s.device, s.dtype
        c_, mu_, sigma_, k_ = c.to(device, dtype), mu.to(device, dtype), sigma.to(device, dtype), k.to(device, dtype)

        # Gaussian falloff (density-like)
        w = torch.exp(-0.5 * ((s - mu_) / sigma_) ** 2)

        # RGB: color mapping
        rgb = c_ * w.unsqueeze(-1)

        # Opacity: optical mapping [0,1]
        opacity = 1.0 - torch.exp(-k_ * w)

        return torch.cat([rgb, opacity.unsqueeze(-1)], dim=-1)

    return tf
