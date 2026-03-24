import os
import numpy as np

from ..common.spatial import apply_uniform_world_transform
from ..core.types import PointCloud


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def point_cloud_xyz_for_export(pc: PointCloud, **kwargs) -> np.ndarray:
    xyz = np.asarray(pc.xyz, dtype=np.float32)
    if kwargs.get("apply_render_world", True):
        transform = kwargs.get("render_world_transform")
        if transform:
            xyz = apply_uniform_world_transform(xyz, transform)
    return xyz
