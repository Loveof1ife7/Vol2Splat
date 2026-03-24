import numpy as np

from ..core.pipeline import Sampler
from ..core.types import Metadata, PointCloud, Volume
from ..registry import register_sampler
from .utils import maybe_to_render_world


class DensitySampler(Sampler):
    def sample(self, vol: Volume, cfg) -> PointCloud:
        n_points = int(cfg.get("n_points", 50000))
        power = float(cfg.get("power", 1.0))
        eps = float(cfg.get("eps", 1e-12))

        if vol.data.ndim == 4:
            scalar = np.asarray(vol.data[3], dtype=np.float32)
        else:
            scalar = np.asarray(vol.data, dtype=np.float32)

        weights = np.clip(scalar.reshape(-1), 0.0, None)
        if power != 1.0:
            weights = np.power(weights, power, dtype=np.float32)
        if not np.isfinite(weights).all() or float(weights.sum()) <= eps:
            weights = np.ones_like(weights, dtype=np.float32)
        probs = weights / weights.sum()

        total = probs.shape[0]
        replace = n_points > total
        linear_indices = np.random.choice(total, size=n_points, replace=replace, p=probs)
        z_idx, y_idx, x_idx = np.unravel_index(linear_indices, scalar.shape)
        ijk = np.column_stack((x_idx, y_idx, z_idx)).astype(np.float32)
        xyz = vol.index_to_world(ijk).astype(np.float32)
        xyz, render_world_applied = maybe_to_render_world(xyz, cfg)

        attrs = {
            "density": scalar.reshape(-1)[linear_indices].reshape(-1, 1).astype(np.float32),
        }
        if vol.data.ndim == 4 and vol.data.shape[0] >= 4:
            rgba = np.stack([vol.data[c].reshape(-1)[linear_indices] for c in range(4)], axis=1).astype(np.float32)
            attrs["rgba"] = rgba
            attrs["rgb"] = rgba[:, :3]
            attrs["opacity"] = rgba[:, 3:4]

        metadata = Metadata(
            source_path=vol.metadata.source_path,
            original_dtype=vol.metadata.original_dtype,
            units=vol.metadata.units,
            extra=dict(vol.metadata.extra),
        )
        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=metadata)
        pc.world_space = "render_world" if render_world_applied else "canonical_world"
        return pc


register_sampler("density", DensitySampler)
