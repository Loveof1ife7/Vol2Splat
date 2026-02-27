
import gc
import torch
import ptwt
from venv import logger
@torch.no_grad()
def dwt3(
    data: torch.Tensor,         # (1, D, H, W) or (4, D, H, W)
    wavelet: str = "haar",
    level: int = 3,
    mode: str = "periodic", # 'reflect'
    multi_channel: bool = False,
    device: str = "cuda"
):
    """
    3D multilevel DWT using ptwt.wavedec3 (GPU version) for radiance data
    Output format matches PyWavelets wavedecn structure:
        [LLL_J, {coeff_J}, {coeff_{J-1}}, ..., {coeff_1}]
    """
    data = data.to(device)
    assert data.ndim == 4, "data must be 4D (C, D, H, W)"

    if not multi_channel:
        assert data.shape[0] == 1, "single_channel data must be (1,D,H,W)"
        if data.numel() > 256 * 256 * 256: 
            coeffs = dwt3_chunked(
                data=data,
                wavelet=wavelet, level=level, mode=mode)
            return coeffs
        else:
            coeffs = ptwt.wavedec3(data, wavelet, level=level, mode=mode)
            return coeffs
    
    if data.ndim == 4 and data.shape[0] == 4:
        rgba = data
    else:
        raise ValueError("multi_channel data must be (4,D,H,W)")

    channels = ["r","g","b","a"]
    out = {}

    for ci, cname in enumerate(channels):
        data_ci = rgba[ci:ci+1, ...] # Keep dim 0 as 1
        if data_ci.numel() > 256 * 256 * 256: 
            logger.info(f"Large data detected, processing channel {cname} in chunks")
            coeffs = dwt3_chunked(data=data_ci,wavelet=wavelet, level=level, mode=mode)
        else:
            coeffs = ptwt.wavedec3(data_ci,wavelet=wavelet, level=level, mode=mode)
            
        out[cname] = coeffs

        del data_ci
        if ci < len(channels) - 1:
            gc.collect()
            torch.cuda.empty_cache()
    return out

@torch.no_grad()
def dwt3_chunked(
     data: torch.Tensor,
     wavelet: str, level: int, mode: str = "periodic",
     chunk_depth: int = 64):
    """
    Fully chunked multi-level 3D DWT.
    - Level 1: chunked on full data
    - Level 2..L: chunked only on LL from previous level
    - No stride-phase mismatch (mathematically correct for Haar/periodic)
    """
    assert wavelet == 'haar' and mode == 'periodic', "dwt_chunked currently only supports wavelet='haar' and mode='periodic'"

    LLL_current = data
    detail_levels = []

    for _ in range(1, level+1):
        LLL_next, details = dwt3_chunked_lvl1(
            data=LLL_current,
            wavelet=wavelet, mode=mode,
            chunk_depth=chunk_depth)
        detail_levels.append(details)
        LLL_current = LLL_next
    
    # Return: [Approximation, Details_Level_N, ..., Details_Level_1]
    return [LLL_current] + detail_levels[::-1]

@torch.no_grad()
def dwt3_chunked_lvl1(data: torch.Tensor, wavelet: str = 'haar', mode: str = "periodic", chunk_depth: int = 64):
    """ 
    DWT level applied with chunking along D dimension.
        Input: data: (1, D, H, W)
        Output: LL_full, details_dict_full
    """
    _, D, H, W = data.shape
    num_chunks = (D + chunk_depth - 1) // chunk_depth
    LLL_chunks, detail_chunks = [], []
    
    for chunk_idx in range(num_chunks):
        D0 = chunk_idx * chunk_depth
        D1 = min((chunk_idx + 1) * chunk_depth, D)
        chunk = data[:, D0:D1]
        
        # Only SINGLE-LEVEL DWT
        coeffs = ptwt.wavedec3(chunk, wavelet=wavelet, level=1, mode=mode)
        
        LLL_chunks.append(coeffs[0])
        detail_chunks.append(coeffs[1])
        
        del chunk
        torch.cuda.empty_cache()

    LLL_full = torch.cat(LLL_chunks, dim=1) # dim 1 is Depth (since input is 1, D, H, W)
    
    detail_full = {}
    if len(detail_chunks) > 0:
        keys = detail_chunks[0].keys()
        for k in keys:
            detail_full[k] = torch.cat([detail_chunks[i][k] for i in range(num_chunks)], dim=1)
            
    return LLL_full, detail_full

@torch.no_grad()
def sparsify_subbands(band, k_budget, lam=3.0):
    """
    band[c] shape: [1, D, H, W]
    Returns:
        idx: torch.int32 (K,3)
        vals: torch.float32 (K,)
        T: float
    """
    if isinstance(band, torch.Tensor):
        device = band.device
        v = band.squeeze(0).abs()
        # If band was (1, D, H, W), squeeze(0) makes it (D, H, W). ndim=3.
        # If band was (4, D, H, W), squeeze(0) makes it (4, D, H, W) if dimension 0 is not 1? No.
        # squeeze(0) only removes dim 0 if size is 1.
        # If input is (4, D, H, W), squeeze(0) does nothing.
        
        if v.ndim == 4:
             # (4, D, H, W) -> Compute norm -> (D, H, W)
             v = torch.norm(v, dim=0)
        
        flat = v.reshape(-1)
        k = min(k_budget, flat.numel())
        vals, topk_idx = torch.topk(flat, k, largest=True)
        D, H, W = v.shape
        z = topk_idx // (H * W)
        y = (topk_idx // W) % H
        x = topk_idx % W
        idx = torch.stack([z, y, x], dim=1).to(torch.int32)
        T = torch.tensor(0.0, device=device, dtype=torch.float32)
        return idx, vals, T
    device = band['a'].device

    v = torch.zeros_like(band['a'].squeeze(0), device=device, dtype=torch.float32)
    for c in ('r', 'g', 'b', 'a'):
        v += (band[c].squeeze(0) ** 2)
    v = torch.sqrt(v + 1e-12)

    flat = v.reshape(-1)

    T = mad_threshold_torch(flat, lam)

    mask = flat > T
    if mask.sum() == 0:
        return (
            torch.empty((0, 3), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            T
        )

    flat_idx = torch.nonzero(mask, as_tuple=False).flatten()
    flat_vals = flat[flat_idx]

    # top-k
    k = min(k_budget, flat_vals.numel())
    vals, topk_idx = torch.topk(flat_vals, k, largest=True)

    keep_flat_idx = flat_idx[topk_idx]

    # unravel idx -> (D,H,W)
    D, H, W = v.shape
    z = keep_flat_idx // (H * W)
    y = (keep_flat_idx // W) % H
    x = keep_flat_idx % W
    idx = torch.stack([z, y, x], dim=1).to(torch.int32)

    return idx, vals, T

def mad_threshold_torch(arr, lam=3.0):
    """Robust threshold using MAD (median absolute deviation), torch version."""
    med = torch.median(arr)
    mad = torch.median(torch.abs(arr - med))
    sigma = mad / 0.6745
    sigma = torch.clamp(sigma, min=torch.finfo(arr.dtype).eps if arr.is_floating_point() else 1e-12)
    return lam * sigma
