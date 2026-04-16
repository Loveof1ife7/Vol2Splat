"""
gs_ply 导出时的坐标修正：与对 xyz 乘以 diag(1,-1,-1) 的手动翻转一致地修正法向与四元数。

T = diag(1,-1,-1) 为绕 X 轴旋转 π 的正交矩阵（det=+1），尺度 log 不变。
"""
import numpy as np

# 与 T = diag(1,-1,-1) 对应的四元数 (w,x,y,z)，满足 R(q_T)=T
_QUAT_NEGATE_YZ_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)


def negate_yz_points(xyz: np.ndarray) -> np.ndarray:
    out = np.array(xyz, dtype=np.float32, copy=True)
    out[:, 1] *= -1.0
    out[:, 2] *= -1.0
    return out


def negate_yz_normals(normals: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """n' = T n，再单位化；零向量保持为零。"""
    n = np.array(normals, dtype=np.float64, copy=True)
    n[:, 1] *= -1.0
    n[:, 2] *= -1.0
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    mask = ln.ravel() > eps
    out = np.zeros_like(n)
    out[mask] = n[mask] / ln[mask]
    return out.astype(np.float32)


def _quat_wxyz_normalize_rows(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.maximum(n, 1e-20)


def quat_mul_wxyz(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Hamilton 积，(w,x,y,z)。qa 可为 (4,) 或 (N,4)，qb 为 (N,4)。"""
    qb = np.asarray(qb, dtype=np.float64).reshape(-1, 4)
    qa = np.asarray(qa, dtype=np.float64).reshape(-1, 4)
    if qa.shape[0] == 1:
        qa = np.broadcast_to(qa, (qb.shape[0], 4))
    assert qa.shape[0] == qb.shape[0], (qa.shape, qb.shape)
    wa, xa, ya, za = qa[:, 0], qa[:, 1], qa[:, 2], qa[:, 3]
    wb, xb, yb, zb = qb[:, 0], qb[:, 1], qb[:, 2], qb[:, 3]
    w = wa * wb - xa * xb - ya * yb - za * zb
    x = wa * xb + xa * wb + ya * zb - za * yb
    y = wa * yb - xa * zb + ya * wb + za * xb
    z = wa * zb + xa * yb - ya * xb + za * wb
    return _quat_wxyz_normalize_rows(np.column_stack([w, x, y, z]))


def negate_yz_rotations(rots_wxyz: np.ndarray) -> np.ndarray:
    """R_new = T @ R_old，等价于 q_new = normalize(q_T ⊗ q_old)。"""
    qb = np.asarray(rots_wxyz, dtype=np.float64).reshape(-1, 4)
    q_new = quat_mul_wxyz(_QUAT_NEGATE_YZ_WXYZ, qb)
    return q_new.astype(np.float32)
