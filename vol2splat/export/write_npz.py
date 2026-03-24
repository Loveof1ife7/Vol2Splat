import numpy as np

from ..core.pipeline import Writer
from ..core.types import PointCloud
from ..registry import register_writer
from .utils import ensure_parent_dir, point_cloud_xyz_for_export


class NPZWriter(Writer):
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        ensure_parent_dir(path)
        arrays = {"xyz": point_cloud_xyz_for_export(pc, **kwargs)}
        for key, value in pc.attrs.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
        np.savez_compressed(path, **arrays)


register_writer("npz", NPZWriter)
