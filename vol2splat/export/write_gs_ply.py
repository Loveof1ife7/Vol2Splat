"""
graphdeco 3D Gaussian Splatting 兼容的 ASCII PLY（属性顺序与 scene/gaussian_model.py 一致）。
opacity / scale 为网络中的**预激活**存储：sigmoid(opacity)、exp(scale)。

export.params：
- negate_yz_axes: 若为 True，对 xyz 做 y,z 取反（同 T=diag(1,-1,-1)），并对法向 n、四元数做一致变换，
  使与手动翻转点云位置后对齐相机时所用的坐标系一致；log-scale 不变。
"""
import numpy as np

from ..core.pipeline import Writer
from ..core.types import PointCloud
from ..registry import register_writer
from .gs_coord_fix import negate_yz_normals, negate_yz_points, negate_yz_rotations
from .utils import ensure_parent_dir, point_cloud_xyz_for_export

_C0 = 0.28209479177387814


def _inverse_sigmoid(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    t = np.clip(x.astype(np.float64), eps, 1.0 - eps)
    return np.log(t / (1.0 - t)).astype(np.float32)


def _rgb_to_f_dc(rgb: np.ndarray) -> np.ndarray:
    """(N,3) linear RGB [0,1] → (N,3) f_dc（与官方 RGB2SH 一致，仅 DC 项）。"""
    c = (rgb.astype(np.float64) - 0.5) / _C0
    return c.astype(np.float32)


class GaussianSplattingPLYWriter(Writer):
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        ensure_parent_dir(path)
        xyz = point_cloud_xyz_for_export(pc, **kwargs).astype(np.float32)
        n_points = xyz.shape[0]

        sh_degree = int(kwargs.get("sh_degree", 3))
        n_rest = 3 * ((sh_degree + 1) ** 2 - 1)

        if "rgb" in pc.attrs:
            rgb = np.asarray(pc.attrs["rgb"], dtype=np.float32).reshape(n_points, -1)[:, :3]
        elif "rgba" in pc.attrs:
            rgb = np.asarray(pc.attrs["rgba"], dtype=np.float32).reshape(n_points, -1)[:, :3]
        else:
            raise ValueError("GaussianSplattingPLYWriter 需要点云 attrs 含 rgb 或 rgba")

        f_dc = _rgb_to_f_dc(np.clip(rgb, 0.0, 1.0))
        f_rest = np.zeros((n_points, n_rest), dtype=np.float32)

        op_lin = np.asarray(pc.attrs.get("opacity", np.full((n_points, 1), 0.1, dtype=np.float32)), dtype=np.float32)
        op_lin = op_lin.reshape(n_points, -1)[:, :1]
        op_lin = np.clip(op_lin, 0.0, 1.0)
        opacity = _inverse_sigmoid(op_lin)

        if all(k in pc.attrs for k in ("scale_0", "scale_1", "scale_2")):
            scales = np.column_stack(
                [
                    np.asarray(pc.attrs["scale_0"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["scale_1"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["scale_2"], dtype=np.float32).reshape(-1),
                ]
            )
        else:
            lin = float(kwargs.get("default_linear_scale", 0.01))
            assert lin > 0, "default_linear_scale must be positive"
            ls = np.log(np.float32(lin))
            scales = np.full((n_points, 3), ls, dtype=np.float32)
        assert scales.shape == (n_points, 3)

        if all(k in pc.attrs for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
            rots = np.column_stack(
                [
                    np.asarray(pc.attrs["rot_0"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["rot_1"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["rot_2"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["rot_3"], dtype=np.float32).reshape(-1),
                ]
            )
        else:
            rots = np.zeros((n_points, 4), dtype=np.float32)
            rots[:, 0] = 1.0
        assert rots.shape == (n_points, 4)

        if all(k in pc.attrs for k in ("nx", "ny", "nz")):
            normals = np.column_stack(
                [
                    np.asarray(pc.attrs["nx"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["ny"], dtype=np.float32).reshape(-1),
                    np.asarray(pc.attrs["nz"], dtype=np.float32).reshape(-1),
                ]
            )
        else:
            normals = np.zeros((n_points, 3), dtype=np.float32)

        negate_yz = bool(kwargs.get("negate_yz_axes", False))
        if negate_yz:
            xyz = negate_yz_points(xyz)
            normals = negate_yz_normals(normals)
            rots = negate_yz_rotations(rots)

        header_lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {n_points}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
        ]
        for i in range(3):
            header_lines.append(f"property float f_dc_{i}")
        for i in range(n_rest):
            header_lines.append(f"property float f_rest_{i}")
        header_lines.append("property float opacity")
        for i in range(3):
            header_lines.append(f"property float scale_{i}")
        for i in range(4):
            header_lines.append(f"property float rot_{i}")
        header_lines.append("end_header\n")

        block = np.hstack(
            [
                xyz,
                normals,
                f_dc,
                f_rest,
                opacity,
                scales,
                rots,
            ]
        )
        fmts = (["%.6f"] * 3 + ["%.6f"] * 3 + ["%.6f"] * 3 + ["%.6f"] * n_rest + ["%.6f"] + ["%.6f"] * 3 + ["%.6f"] * 4)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
            np.savetxt(f, block, fmt=fmts)


register_writer("gs_ply", GaussianSplattingPLYWriter)
