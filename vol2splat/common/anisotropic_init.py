"""
体素标量场梯度 → 世界空间法线、3DGS 四元数（局部 +Z 对齐法线）、各向异性尺度（log 空间）。
"""
import numpy as np

from ..core.types import Volume


def _characteristic_voxel_size(vol: Volume, mode: str) -> float:
    sp = np.asarray(vol.spacing, dtype=np.float64).ravel()
    assert sp.shape == (3,)
    if mode == "max_spacing":
        return float(np.max(sp))
    if mode == "mean_spacing":
        return float(np.mean(sp))
    raise ValueError(f"Unknown aniso_voxel_size_mode: {mode}")


def scalar_gradient_index_to_world(vol: Volume, scalar_zyx: np.ndarray) -> np.ndarray:
    """
    对 (Z,Y,X) 标量场求梯度，并映射到世界坐标系下的 ∇D（行向量与 xyz 对齐）。

    index_to_world: world_row = ijk_row * spacing ⊙ direction.T（与 Volume.index_to_world 一致）。
    链式法则给出：g_world_row = g_ijk_row @ vol.direction @ diag(1/spacing)。
    """
    assert scalar_zyx.ndim == 3
    gz, gy, gx = np.gradient(np.asarray(scalar_zyx, dtype=np.float64))
    # 最后一维顺序为 ∂/∂x, ∂/∂y, ∂/∂z（与 ijk 的 x,y,z 一致）
    g_ijk = np.stack([gx, gy, gz], axis=-1)
    R = np.asarray(vol.direction, dtype=np.float64)
    sp = np.asarray(vol.spacing, dtype=np.float64).ravel()
    inv_s = 1.0 / sp
    # g_world[z,y,x,:] = g_ijk[z,y,x,:] @ R @ diag(inv_s)
    M = R * inv_s[np.newaxis, :]
    return np.einsum("...i,ij->...j", g_ijk, M).astype(np.float32)


def quat_wxyz_rotate_pos_z_to_v_batch(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N,3) 向量 → (N,4) 单位四元数 w,x,y,z：局部 +Z 旋转到与各行 v 同向。"""
    v = np.asarray(v, dtype=np.float64)
    assert v.ndim == 2 and v.shape[1] == 3
    n_pts = v.shape[0]
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    nv = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.where(nv >= eps, v / np.maximum(nv, eps), np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    dot = np.clip(np.sum(z * n, axis=1), -1.0, 1.0)
    cross = np.column_stack([-n[:, 1], n[:, 0], np.zeros(n_pts, dtype=np.float64)])
    nc = np.linalg.norm(cross, axis=1)
    w = 1.0 + dot
    q = np.column_stack([w, cross[:, 0], cross[:, 1], cross[:, 2]])
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(qn, eps)

    out = q.copy()
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    low_nv = nv.ravel() < eps
    out[low_nv] = identity
    parallel = (nc < eps) & (~low_nv)
    out[parallel & (dot > 0)] = identity
    out[parallel & (dot <= 0)] = flip
    return out.astype(np.float32)


def gather_zyx(arr_zyx: np.ndarray, z_idx: np.ndarray, y_idx: np.ndarray, x_idx: np.ndarray) -> np.ndarray:
    return arr_zyx[z_idx, y_idx, x_idx]


def anisotropic_scales_and_rots(
    vol: Volume,
    grad_world_zyx: np.ndarray,
    z_idx: np.ndarray,
    y_idx: np.ndarray,
    x_idx: np.ndarray,
    tangent_mul: float,
    normal_mul: float,
    voxel_size_mode: str,
    grad_eps: float,
    min_linear_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      log_scales (N,3) — 3DGS 内部 log(exp)=线性轴长，顺序为局部 x,y,z（Z 为法向薄方向）
      rots (N,4) — w,x,y,z
      normals (N,3) — 单位法向（与 rots 中局部 +Z 对齐），用于写入 PLY nx,ny,nz
    """
    g = gather_zyx(grad_world_zyx, z_idx, y_idx, x_idx)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    fallback = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    n = np.where(norms > grad_eps, (-g / np.maximum(norms, grad_eps)).astype(np.float32), fallback)

    voxel = _characteristic_voxel_size(vol, voxel_size_mode)
    s_t = float(voxel * tangent_mul)
    s_n = float(voxel * normal_mul)
    s_t = max(s_t, min_linear_scale)
    s_n = max(s_n, min_linear_scale)

    rots = quat_wxyz_rotate_pos_z_to_v_batch(n.astype(np.float64))
    n_pts = int(z_idx.shape[0])
    log_scales = np.empty((n_pts, 3), dtype=np.float32)
    log_scales[:, 0] = np.log(s_t)
    log_scales[:, 1] = np.log(s_t)
    log_scales[:, 2] = np.log(s_n)
    return log_scales, rots, n.astype(np.float32, copy=False)


def apply_uniform_scale_to_log_scales(log_scales: np.ndarray, scale_factor: float) -> None:
    if abs(scale_factor - 1.0) < 1e-12:
        return
    log_scales[:, :] = log_scales + np.log(float(scale_factor))
