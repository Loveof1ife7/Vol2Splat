import numpy as np
from typing import Dict, Any
from ..core.types import Volume, PointCloud
from ..core.pipeline import Sampler
from ..registry import register_sampler

class GradientSampler(Sampler):
    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        n_points = cfg.get("n_points", 100000)
        
        # Volume shape is (Z, Y, X)
        shape = vol.shape 
        data = vol.data.astype(np.float32)

        # Compute gradients
        gz, gy, gx = np.gradient(data)
        mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # argsort ON log N, but we can do partial sort for top K
        mag_flat = mag.flatten()
        top_indices = np.argpartition(mag_flat, -n_points)[-n_points:]
        z_idx, y_idx, x_idx = np.unravel_index(top_indices, shape)
        ijk = np.column_stack((x_idx, y_idx, z_idx))

        # ijk to world coords
        xyz = vol.index_to_world(ijk)

        # normal vector
        intensities = data.ravel()[top_indices].reshape(-1, 1)
        grad_vectors = np.column_stack((
            gx.ravel()[top_indices], 
            gy.ravel()[top_indices], 
            gz.ravel()[top_indices]
        ))
        mags = mag_flat[top_indices].reshape(-1, 1) + 1e-8
        normals = grad_vectors / mags  
        
        attrs = {
            "intensity": intensities,
            "normal": normals,       
            "gradient_mag": mags     
        }
        
        return PointCloud(xyz=xyz.astype(np.float32), attrs=attrs)

register_sampler("gradient", GradientSampler)


