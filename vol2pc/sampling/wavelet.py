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
    Density-field Wavelet Sampler.

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

        # 1. to tensor: (1, Z, Y, X) on device 
        data_np = vol.data.astype(np.float32, copy=False)
        if data_np.ndim != 3:
            raise ValueError(f"WaveletSampler expects density volume (Z,Y,X), got {data_np.shape}")

        data_tensor = torch.from_numpy(data_np).unsqueeze(0).contiguous().to(device)  # (1,D,H,W)

        # 2. DWT 
        coeffs = self.wavelet_transform(
            data=data_tensor,
            wavelet=wavelet,
            level=level,
            mode=mode,
            multi_channel=False,
            device=device
        )

        # 3. analyze bands and allocate k for each band
        shape_full = torch.tensor(vol.data.shape, device=device, dtype=torch.long)  # (Z,Y,X)
        bands_info = self.bands_info_from_layout(layout=coeffs, level=level)

        k_alloc = self.alloc_k_for_bands(
            bands_info=bands_info,
            K_total=n_points,
            alpha=alpha,
            beta=beta,
            k_min=k_min
        )

        # anchor center in zyx
        Z, Y, X = vol.data.shape
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
            # You may choose to skip approx (lev==0) in step1, but keep it for now.
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
            "band": np.array(all_band, dtype=object),
        }

        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=vol.metadata)
        return pc

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
        for lev, dic in enumerate(layout):
            if lev == 0:
                assert isinstance(dic, torch.Tensor), "layout[0] (approx) is not a tensor"
                o = "aaa"
                band_ref = dic
                E = torch.sum(band_ref ** 2)
                N = int(band_ref.numel())
                j = 0  # zero wavelet level refer to the 'aaa'('LLL') band
                bands_info.append((lev, j, o, band_ref, E, N))
                continue

            for o, band_ref in dic.items():
                E = torch.sum(band_ref ** 2)
                N = int(band_ref.numel())
                j = level - (lev - 1)  # matches your test: lev=1 -> j=level, lev=2 -> j=level-1 ...
                bands_info.append((lev, j, o, band_ref, E, N))
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
