import numpy as np

from ..common.tf import bake_rgba_volume
from ..core.pipeline import Sampler
from ..core.types import Metadata, PointCloud, Volume
from ..io.read_vti import VTIReader
from ..registry import register_sampler
from .utils import maybe_to_render_world, resolve_tf_json_path


class OpacitySampler(Sampler):
    def sample_canonical_vti(self, canonical_vti_path: str, cfg):
        vol = VTIReader().read(canonical_vti_path)
        return self.sample(vol, cfg)

    def sample(self, vol: Volume, cfg) -> PointCloud:
        n_points = int(cfg.get("n_points", 50000))
        tf_backend = cfg.get("tf_backend", "auto")
        tf_device = cfg.get("tf_device")
        premultiply_alpha = bool(cfg.get("premultiply_alpha", False))

        if vol.data.ndim == 4 and vol.data.shape[0] == 4:
            rgba = np.asarray(vol.data, dtype=np.float32)
        else:
            tf_json = resolve_tf_json_path(cfg)
            rgba = bake_rgba_volume(
                vol.data,
                tf_json_path=tf_json,
                premultiply_alpha=premultiply_alpha,
                backend=tf_backend,
                device=tf_device,
                return_numpy=True,
            )

        alpha = np.asarray(rgba[3], dtype=np.float32)
        probs = np.clip(alpha.reshape(-1), 0.0, None)
        if probs.sum() <= 1e-12:
            probs = np.ones_like(probs, dtype=np.float32)
        probs = probs / probs.sum()

        total = probs.shape[0]
        replace = n_points > total
        indices = np.random.choice(total, size=n_points, replace=replace, p=probs)
        z_idx, y_idx, x_idx = np.unravel_index(indices, alpha.shape)
        ijk = np.column_stack((x_idx, y_idx, z_idx)).astype(np.float32)
        xyz = vol.index_to_world(ijk).astype(np.float32)
        xyz, render_world_applied = maybe_to_render_world(xyz, cfg)

        scalar = np.asarray(vol.data if vol.data.ndim == 3 else rgba[3], dtype=np.float32)
        attrs = {
            "opacity": alpha.reshape(-1)[indices].reshape(-1, 1),
            "rgb": np.stack(
                [
                    rgba[0].reshape(-1)[indices],
                    rgba[1].reshape(-1)[indices],
                    rgba[2].reshape(-1)[indices],
                ],
                axis=1,
            ).astype(np.float32),
            "rgba": np.stack(
                [
                    rgba[0].reshape(-1)[indices],
                    rgba[1].reshape(-1)[indices],
                    rgba[2].reshape(-1)[indices],
                    rgba[3].reshape(-1)[indices],
                ],
                axis=1,
            ).astype(np.float32),
            "density": scalar.reshape(-1)[indices].reshape(-1, 1).astype(np.float32),
        }
        metadata = Metadata(
            source_path=vol.metadata.source_path,
            original_dtype=vol.metadata.original_dtype,
            units=vol.metadata.units,
            extra=dict(vol.metadata.extra),
        )
        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=metadata)
        pc.world_space = "render_world" if render_world_applied else "canonical_world"
        return pc


register_sampler("opacity", OpacitySampler)
