import numpy as np
import torch
import ptwt
import gc
from typing import Dict, Any, Tuple, Union, List
from .low_level.ll_wavelet import dwt3, sparsify_subbands
from .utils import to_torch
from ..core.types import Volume, PointCloud
from ..core.pipeline import Sampler
from ..registry import register_sampler
from ..core.log import get_logger

logger = get_logger("sampling.wavelet")

class WaveletSampler(Sampler):
    """
    Density/RGBA-field Wavelet Sampler.

    Assumptions:
        - vol.data is a scalar density volume with shape (Z, Y, X)
        - We convert to torch tensor (1, D, H, W) where D=Z, H=Y, W=X
        - dwt3 returns: [approx, details_level_L, ..., details_level_1]
          where each details_level_* is a dict: band_name -> tensor (1, Dz, Dy, Dx)
    """

    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        n_points = int(cfg.get("n_points", 10000))
        wavelet  = cfg.get("wavelet", "haar")
        level    = int(cfg.get("level", 1))
        mode     = cfg.get("mode", "periodic")
        alpha    = float(cfg.get("alpha", 1.0))
        beta     = float(cfg.get("beta", 1.0))
        k_min    = int(cfg.get("k_min", 100))
        device   = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Wavelet sampling: {wavelet} level={level} points={n_points} on {device}")

        # 1. to tensor: (C, Z, Y, X) on device
        data_np = vol.data.astype(np.float32, copy=False)
        data_tensor, multi_channel = self.shape_check_and_to_tensor(data_np, device)

        # 2. DWT 
        coeffs = self.wavelet_transform(
            data=data_tensor,
            wavelet=wavelet,
            level=level,
            mode=mode,
            multi_channel=multi_channel,
            device=device
        )
        
        # If multi_channel, coeffs is a dict {'r': [...], 'g': [...], ...}
        # We need to unify this layout to reuse alloc_k_for_bands logic.
        # Strategy:
        # If multi-channel, we can process each channel separately or aggregate energy.
        # Aggregating energy seems appropriate for sampling "structure" across all channels.
        
        layout = []
        if multi_channel:
             # coeffs is a dict: {'r': [approx, details...], 'g': ...}
             # We want to merge them into a single structure where each band tensor is (4, D, H, W) or we aggregate energy now.
             # alloc_k_for_bands expects a list [approx, dict(details), ...] 
             # where approx is tensor and details is dict {band_name: tensor}
             # The structure of coeffs['r'] is: [approx_tensor, {details_level_L}, ...]
             # We will reconstruct a "virtual" single channel layout where each tensor has 4 channels
             # and we sum energy over channels in bands_info_from_layout.
             # Note: vol.shape is (C, Z, Y, X) for 4D volume, but VolSampler Volume object usually has data.shape
             # However, VolSampler assumes (Z, Y, X) usually. For 4D data it's (C, Z, Y, X).
             # Let's trust data_np.shape logic from shape_check_and_to_tensor
             
             # C, X, Y, Z = vol.shape # This is risky if vol.shape is just data.shape
             
             # Let's get dims from tensor
             # data_tensor is (C, Z, Y, X)
             Z, Y, X = data_tensor.shape[1], data_tensor.shape[2], data_tensor.shape[3]
             
             channels = ['r', 'g', 'b', 'a']
             # Check consistency
             n_levels = len(coeffs['r'])
             
             # 1. Approx level (index 0)
             # Stack [1, D, H, W] -> [4, D, H, W]
             approx_list = [coeffs[c][0] for c in channels]
             approx_combined = torch.cat(approx_list, dim=0)
             layout.append(approx_combined)
             
             # 2. Detail levels
             for l in range(1, n_levels):
                  # coeffs['r'][l] is a dict {'aad': tensor, ...}
                  combined_details = {}
                  keys = coeffs['r'][l].keys()
                  for k in keys:
                       # Stack
                       tensor_list = [coeffs[c][l][k] for c in channels]
                       combined_details[k] = torch.cat(tensor_list, dim=0)
                  layout.append(combined_details)
        else:
            # Single channel: data_tensor is (1, Z, Y, X)
            Z, Y, X = data_tensor.shape[1], data_tensor.shape[2], data_tensor.shape[3]
            layout = coeffs

        # 3. analyze bands and allocate k for each band
        shape_full = torch.tensor([Z, Y, X], device=device, dtype=torch.long)
        bands_info = self.bands_info_from_layout(layout=layout, level=level)

        k_alloc = self.alloc_k_for_bands(
            bands_info=bands_info,
            K_total=n_points,
            alpha=alpha,
            beta=beta,
            k_min=k_min
        )

        # anchor center in zyx
        mu0_anchor = torch.tensor([(Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0],
                                  device=device, dtype=torch.float32)

        # 4. sparsify each band -> centers in full-res zyx
        all_centers = []
        all_vals = []
        all_level = []
        all_band = []

        for (lev, j, o, band_ref, E, N), k_budget in zip(bands_info, k_alloc):
            k_budget = int(k_budget.item()) if torch.is_tensor(k_budget) else int(k_budget)
            if k_budget <= 0:
                continue

            # band_ref: torch.Tensor (1, Dz, Dy, Dx) for both approx and detail
            # Or (4, Dz, Dy, Dx) if aggregated in step 2.
            # sparsify_subbands handles (4, ...) by computing norm.
            
            idx, vals, T = sparsify_subbands(band_ref, k_budget)  # idx: (K,3) zyx in subband grid

            if idx is None or idx.numel() == 0:
                continue

            centers = self.coeff_index_to_center_gpu(
                idxs=idx,
                j=j,
                level=level,
                shape=shape_full,         # IMPORTANT: full shape
                mu0_anchor=mu0_anchor,
                device=device
            )  # (K,3) zyx in full-res (float)

            all_centers.append(centers)
            all_vals.append(vals)
            all_level.append(torch.full((centers.shape[0],), j, device=device, dtype=torch.int32))
            all_band += [o] * centers.shape[0]

        if not all_centers:
            return PointCloud(xyz=np.zeros((0, 3), np.float32), attrs={}, metadata=vol.metadata)

        centers_zyx = torch.cat(all_centers, dim=0)   # (K,3) float
        vals = torch.cat(all_vals, dim=0)             # (K,)
        levels = torch.cat(all_level, dim=0)          # (K,)

        # 5. cap to n_points by coeff magnitude 
        if vals.numel() > n_points:
            vv = vals.abs()
            topv, topi = torch.topk(vv, n_points, largest=True)
            centers_zyx = centers_zyx[topi]
            vals = vals[topi]
            levels = levels[topi]
            all_band = [all_band[i] for i in topi.tolist()]

        # 6. centers -> world xyz 
        centers_np = centers_zyx.detach().cpu().numpy()
        z = centers_np[:, 0]
        y = centers_np[:, 1]
        x = centers_np[:, 2]

        # Volume.index_to_world expects ijk=(i,j,k)=(x,y,z)
        ijk = np.stack([x, y, z], axis=1).astype(np.float64)
        xyz = vol.index_to_world(ijk).astype(np.float32)

        attrs = {
            "coeff_abs": vals.abs().detach().cpu().numpy().astype(np.float32),
            "level": levels.detach().cpu().numpy().astype(np.int32),
            # "band": np.array(all_band, dtype=object), # skip string attr for PLY safety
        }
        
        # Add RGBA color attribute if input was RGBA
        if multi_channel:
             # We need to sample the RGBA volume at the selected coordinates
             # centers_zyx is (K, 3) float coordinates in index space (z, y, x)
             # vol.data is (4, Z, Y, X)
             
             # Use grid_sample for interpolation
             # grid_sample expects input (N, C, D, H, W) and grid (N, D, H, W, 3) in [-1, 1] range
             # Our centers are in [0, Z-1], [0, Y-1], [0, X-1]
             
             # Normalize coordinates to [-1, 1]
             # (2 * coord / (size - 1)) - 1
             
             K = centers_zyx.shape[0]
             
             # Prepare grid: (1, K, 1, 1, 3) for sampling K points?
             # Or just (1, 1, 1, K, 3)?
             # PyTorch grid_sample is for dense grids usually.
             # For sparse points, we can reshape them to (1, 1, 1, K, 3)
             
             # Coordinates order for grid_sample is (x, y, z)
             # centers_zyx is (z, y, x)
             # So we need (centers_x, centers_y, centers_z)
             
             cx = centers_zyx[:, 2]
             cy = centers_zyx[:, 1]
             cz = centers_zyx[:, 0]
             
             # Normalize
             # X, Y, Z are spatial dimensions
             # grid_sample expects coordinates in [-1, 1]
             # index 0 -> -1, index size-1 -> 1
             # formula: 2 * (index / (size - 1)) - 1
             # If size is 1 (unlikely for spatial dims but possible), this is nan.
             
             def normalize_coord(c, size):
                 if size > 1:
                     return (2 * c / (size - 1)) - 1
                 return torch.zeros_like(c)

             nx = normalize_coord(cx, X)
             ny = normalize_coord(cy, Y)
             nz = normalize_coord(cz, Z)
             
             grid = torch.stack([nx, ny, nz], dim=1) # (K, 3)
             grid = grid.view(1, 1, 1, K, 3) # (N, Dout, Hout, Wout, 3)
             
             # Input data
             input_tensor = data_tensor # (4, Z, Y, X)
             if input_tensor.ndim == 4:
                  input_tensor = input_tensor.unsqueeze(0) # (1, 4, Z, Y, X)
             
             # Sample
             # align_corners=True matches the -1 to 1 mapping with boundary pixels
             sampled_rgba = torch.nn.functional.grid_sample(
                  input_tensor, 
                  grid, 
                  mode='nearest', # Use nearest to avoid interpolation artifacts or out of bounds
                  padding_mode='border', 
                  align_corners=True
             ) # (1, 4, 1, 1, K)
             
             # Reshape to (K, 4)
             sampled_rgba = sampled_rgba.view(4, K).permute(1, 0) # (K, 4)
             
             # Add to attributes
             rgba_np = sampled_rgba.detach().cpu().numpy().astype(np.float32)
             
             # Add individual channels for PLY writer convenience (if it doesn't handle vector attrs well, though PLYWriter usually does)
             # PLYWriter handles (K, 4) if key is like "colors" or we can split.
             # Let's add as separate attributes or standard ply colors?
             # Standard PLY colors are usually uchar 0-255 named red, green, blue, alpha.
             # Or float named r, g, b, a?
             # Let's add as 'red', 'green', 'blue', 'alpha' float properties.
             
             attrs['red'] = rgba_np[:, 0:1]
             attrs['green'] = rgba_np[:, 1:2]
             attrs['blue'] = rgba_np[:, 2:3]
             attrs['alpha'] = rgba_np[:, 3:4]

        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=vol.metadata)
        return pc

    def shape_check_and_to_tensor(self, data_np, device):
        data_tensor = None
        multi_channel = False
        if data_np.ndim == 3:
             # (Z, Y, X) -> (1, Z, Y, X)
             data_tensor = torch.from_numpy(data_np).unsqueeze(0).contiguous().to(device)
             multi_channel = False
        elif data_np.ndim == 4:
             # (C, Z, Y, X) -> (C, Z, Y, X)
             # Assume C=4 for RGBA,
             # dwt3 handles multi_channel if multi_channel=True and input is (4, D, H, W)
             if data_np.shape[0] == 4:
                  data_tensor = torch.from_numpy(data_np).contiguous().to(device)
                  multi_channel = True
             else:
                  # If C != 4 but ndim=4, it's (1, Z, Y, X) 
                  if data_np.shape[0] == 1:
                       data_tensor = torch.from_numpy(data_np).contiguous().to(device)
                       multi_channel = False
                  else:
                       # For now, let's stick to supporting 1 or 4 channels as per user request (RGBA).
                       raise ValueError(f"WaveletSampler supports 1 or 4 channels, got {data_np.shape[0]}")
        else:
             raise ValueError(f"WaveletSampler expects volume (Z,Y,X) or (4,Z,Y,X), got {data_np.shape}")
        return data_tensor, multi_channel
    
    @torch.no_grad()
    def wavelet_transform(self, data: torch.Tensor, wavelet: str, level: int, mode: str, multi_channel: bool, device: str):
        return dwt3(
            data=data,
            wavelet=wavelet,
            level=level,
            mode=mode,
            multi_channel=multi_channel,
            device=device
        )

    def bands_info_from_layout(self, layout, level):
        bands_info = []
        for lev, bands_dic in enumerate(layout):
            if lev == 0:
                assert isinstance(bands_dic, torch.Tensor), "layout[0] (approx) is not a tensor"
                o = "aaa"
                band_ref = bands_dic
                E = torch.sum(band_ref ** 2)
                N = int(band_ref.numel())
                j = 0  # zero wavelet level refer to the 'aaa'('LLL') band
                bands_info.append((lev, j, o, band_ref, E, N))
                continue

            for o, band in bands_dic.items():
                E = torch.sum(band ** 2)
                N = int(band.numel())
                j = level - (lev - 1)  # matches your test: lev=1 -> j=level, lev=2 -> j=level-1 ...
                bands_info.append((lev, j, o, band, E, N))
        return bands_info

    def alloc_k_for_bands(self, bands_info, K_total, alpha=1.0, beta=1.0, gamma=1.2, k_min=100, mode="default"):
        device = bands_info[0][3].device
        k_alloc = None

        if mode == "default":
            energies = torch.stack([b[4] for b in bands_info]).to(device=device, dtype=torch.float32)
            sizes    = torch.tensor([b[5] for b in bands_info], device=device, dtype=torch.float32)
            j        = torch.tensor([b[1] for b in bands_info], device=device, dtype=torch.float32)

            E_smooth = torch.log1p(energies)
            Lw = (j + 1.0) ** (-gamma)
            scores = (E_smooth ** alpha) * (sizes ** beta) * Lw
            weights = scores / (scores.sum() + 1e-12)

            k_alloc = torch.floor(weights * float(K_total)).clamp(min=k_min).to(torch.int32)
        else:
            energies = torch.stack([b[4] for b in bands_info]).to(device=device, dtype=torch.float32)
            sizes = torch.tensor([b[5] for b in bands_info], device=device, dtype=torch.float32)
            scores = (energies ** alpha) * (sizes ** beta)
            weights = scores / (scores.sum() + 1e-12)
            k_alloc = torch.floor(float(K_total) * weights).clamp(min=max(1, k_min)).to(torch.int32)

        return k_alloc

    def coeff_index_to_center_gpu(self, idxs, j, level, shape, mu0_anchor=None, axis_order="zyx", device=None):
        if device is None:
            device = idxs.device
        if j == 0: 
            j = level # zero wavelet level refer to the 'aaa'('LLL') band

        base = to_torch(shape, dtype=torch.long, device=device)  # (Z,Y,X) full
        dj = base // (2 ** j)                   # expected subband size
        center_sub = dj * 0.5

        idxs = idxs.to(torch.float32)
        center_sub = center_sub.to(torch.float32)
        Delta = (idxs - center_sub) * (2 ** j)

        mu0 = to_torch(mu0_anchor, dtype=torch.float32, device=device)
        centers_zyx = mu0[None, :] + Delta

        if axis_order == "zyx":
            return centers_zyx
        raise ValueError("only support 'zyx' axis_order")
    
     
register_sampler("wavelet", WaveletSampler)
