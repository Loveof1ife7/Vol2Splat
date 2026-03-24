import numpy as np

from ..core.pipeline import Sampler
from ..core.types import PointCloud, Volume
from ..registry import register_sampler


class PoissonSampler(Sampler):
    def sample(self, vol: Volume, cfg) -> PointCloud:
        n_points = int(cfg.get("n_points", 10000))
        radius = float(cfg.get("radius", 0.0))
        oversample = int(cfg.get("oversample", max(4, min(16, n_points // 1000 + 4))))

        shape = vol.shape[-3:]
        candidate_count = max(n_points * oversample, n_points)
        x_idx = np.random.uniform(0, shape[2] - 1, candidate_count)
        y_idx = np.random.uniform(0, shape[1] - 1, candidate_count)
        z_idx = np.random.uniform(0, shape[0] - 1, candidate_count)
        ijk = np.stack([x_idx, y_idx, z_idx], axis=1)
        xyz = vol.index_to_world(ijk).astype(np.float32)

        if radius <= 0:
            radius = 0.5 * max(shape) / max(n_points ** (1.0 / 3.0), 1.0)

        chosen = []
        radius2 = radius * radius
        for idx, point in enumerate(xyz):
            if len(chosen) >= n_points:
                break
            keep = True
            for picked_idx in chosen:
                delta = point - xyz[picked_idx]
                if float(np.dot(delta, delta)) < radius2:
                    keep = False
                    break
            if keep:
                chosen.append(idx)

        if len(chosen) < n_points:
            missing = n_points - len(chosen)
            remaining = np.setdiff1d(np.arange(candidate_count), np.asarray(chosen, dtype=np.int64), assume_unique=False)
            if remaining.size > 0:
                extra = remaining[:missing]
                chosen.extend(extra.tolist())

        chosen = np.asarray(chosen[:n_points], dtype=np.int64)
        xi = np.clip(np.round(x_idx[chosen]).astype(int), 0, shape[2] - 1)
        yi = np.clip(np.round(y_idx[chosen]).astype(int), 0, shape[1] - 1)
        zi = np.clip(np.round(z_idx[chosen]).astype(int), 0, shape[0] - 1)
        scalar = vol.data[zi, yi, xi] if vol.data.ndim == 3 else vol.data[3, zi, yi, xi]

        return PointCloud(
            xyz=xyz[chosen],
            attrs={"density": np.asarray(scalar, dtype=np.float32).reshape(-1, 1)},
            metadata=vol.metadata,
        )


register_sampler("poisson", PoissonSampler)
