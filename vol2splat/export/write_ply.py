import numpy as np

from ..core.pipeline import Writer
from ..core.types import PointCloud
from ..registry import register_writer
from .utils import ensure_parent_dir, point_cloud_xyz_for_export


class PLYWriter(Writer):
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        ensure_parent_dir(path)
        xyz = point_cloud_xyz_for_export(pc, **kwargs)
        n_points = xyz.shape[0]

        attr_specs = []
        for key in sorted(pc.attrs.keys()):
            arr = pc.attrs[key]
            if not isinstance(arr, np.ndarray) or not np.issubdtype(arr.dtype, np.number):
                continue
            arr = arr.reshape(n_points, -1).astype(np.float32, copy=False)
            if arr.shape[1] == 1:
                names = [key]
            else:
                names = [f"{key}_{idx}" for idx in range(arr.shape[1])]
            attr_specs.append((names, arr))

        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            for names, _ in attr_specs:
                for name in names:
                    f.write(f"property float {name}\n")
            f.write("end_header\n")

            blocks = [xyz.astype(np.float32, copy=False)]
            for _, arr in attr_specs:
                blocks.append(arr)
            data = np.hstack(blocks)
            np.savetxt(f, data, fmt="%.6f")


register_writer("ply", PLYWriter)
