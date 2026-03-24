import numpy as np

from ..core.pipeline import Sampler
from ..core.types import PointCloud, Volume
from ..registry import register_sampler
from .gradient import GradientSampler
from .uniform import UniformSampler


class HybridSampler(Sampler):
    def sample(self, vol: Volume, cfg) -> PointCloud:
        n_points = int(cfg.get("n_points", 50000))
        gradient_ratio = float(cfg.get("gradient_ratio", 0.7))
        n_gradient = max(0, min(n_points, int(round(n_points * gradient_ratio))))
        n_uniform = max(0, n_points - n_gradient)

        pcs = []
        if n_gradient > 0:
            pcs.append(GradientSampler().sample(vol, {**cfg, "n_points": n_gradient}))
        if n_uniform > 0:
            pcs.append(UniformSampler().sample(vol, {**cfg, "n_points": n_uniform}))

        xyz = np.concatenate([pc.xyz for pc in pcs], axis=0) if pcs else np.empty((0, 3), dtype=np.float32)
        attr_keys = sorted({key for pc in pcs for key in pc.attrs.keys()})
        attrs = {}
        for key in attr_keys:
            parts = []
            width = None
            for pc in pcs:
                arr = pc.attrs.get(key)
                if arr is not None:
                    arr = arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)
                    width = arr.shape[1]
                    parts.append(arr)
                else:
                    if width is None:
                        continue
                    parts.append(np.zeros((pc.xyz.shape[0], width), dtype=np.float32))
            if parts:
                attrs[key] = np.concatenate(parts, axis=0)

        return PointCloud(xyz=xyz.astype(np.float32, copy=False), attrs=attrs, metadata=vol.metadata)


register_sampler("hybrid", HybridSampler)
