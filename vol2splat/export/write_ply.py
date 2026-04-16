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

        rgb_u8 = None
        if "rgb" in pc.attrs and isinstance(pc.attrs["rgb"], np.ndarray):
            rgb_arr = pc.attrs["rgb"].reshape(n_points, -1)
            assert rgb_arr.shape[1] >= 3, "rgb attribute must have at least 3 channels"
            rgb_u8 = np.clip(rgb_arr[:, :3], 0.0, 1.0)
            rgb_u8 = (rgb_u8 * 255.0 + 0.5).astype(np.uint8)
        elif "rgba" in pc.attrs and isinstance(pc.attrs["rgba"], np.ndarray):
            rgba_arr = pc.attrs["rgba"].reshape(n_points, -1)
            assert rgba_arr.shape[1] >= 3, "rgba attribute must have at least 3 channels"
            rgb_u8 = np.clip(rgba_arr[:, :3], 0.0, 1.0)
            rgb_u8 = (rgb_u8 * 255.0 + 0.5).astype(np.uint8)

        attr_specs = []
        for key in sorted(pc.attrs.keys()):
            if key in {"rgb", "rgba"}:
                continue
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
            if rgb_u8 is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            for names, _ in attr_specs:
                for name in names:
                    f.write(f"property float {name}\n")
            f.write("end_header\n")

            blocks = [xyz.astype(np.float32, copy=False)]
            fmts = ["%.6f", "%.6f", "%.6f"]
            if rgb_u8 is not None:
                blocks.append(rgb_u8.astype(np.float32, copy=False))
                fmts.extend(["%d", "%d", "%d"])
            for _, arr in attr_specs:
                blocks.append(arr)
                fmts.extend(["%.6f"] * arr.shape[1])
            data = np.hstack(blocks)
            np.savetxt(f, data, fmt=fmts)


register_writer("ply", PLYWriter)
