import numpy as np
from typing import Dict, Any
from ..core.types import Metadata, Volume, PointCloud
from ..core.pipeline import Sampler
from ..registry import register_sampler
from .utils import maybe_to_render_world

class UniformSampler(Sampler):
    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        n_points = cfg.get("n_points", 1000)
        
        # Volume shape is (Z, Y, X)
        shape = vol.shape 
        
        # Sample random indices in continuous space
        # x index range [0, X-1]
        x_idx = np.random.uniform(0, shape[2]-1, n_points)
        y_idx = np.random.uniform(0, shape[1]-1, n_points)
        z_idx = np.random.uniform(0, shape[0]-1, n_points)
        
        # Construct ijk for world conversion: (x, y, z) order
        ijk = np.stack([x_idx, y_idx, z_idx], axis=1)
        
        # Convert to world coords
        xyz = vol.index_to_world(ijk)
        xyz, render_world_applied = maybe_to_render_world(xyz, cfg)
        
        # Sample values (Nearest Neighbor)
        xi = np.round(x_idx).astype(int)
        yi = np.round(y_idx).astype(int)
        zi = np.round(z_idx).astype(int)
        
        # Clip
        xi = np.clip(xi, 0, shape[2]-1)
        yi = np.clip(yi, 0, shape[1]-1)
        zi = np.clip(zi, 0, shape[0]-1)
        
        # Access data: vol.data[z, y, x]
        values = vol.data[zi, yi, xi]
        
        attrs = {"v": values.reshape(-1, 1)}
        
        metadata = Metadata(
            source_path=vol.metadata.source_path,
            original_dtype=vol.metadata.original_dtype,
            units=vol.metadata.units,
            extra=dict(vol.metadata.extra),
        )
        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=metadata)
        pc.world_space = "render_world" if render_world_applied else "canonical_world"
        return pc

register_sampler("uniform", UniformSampler)
