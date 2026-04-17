import numpy as np

from ...common.anisotropic_init import (
    anisotropic_scales_and_rots,
    apply_uniform_scale_to_log_scales,
    scalar_gradient_index_to_world,
)
from ..utils import resolve_render_world_transform


def sample_indices_with_uniform_tf_filter(
    alpha: np.ndarray,
    n_points: int,
    alpha_threshold: float,
    uniform_stride_cfg,
) -> np.ndarray:
    flat_alpha = np.clip(np.asarray(alpha, dtype=np.float32).reshape(-1), 0.0, None)
    total = int(flat_alpha.shape[0])
    z_dim, y_dim, x_dim = alpha.shape

    uniform_stride = _resolve_uniform_stride(
        alpha_shape=alpha.shape,
        total=total,
        n_points=n_points,
        flat_alpha=flat_alpha,
        alpha_threshold=alpha_threshold,
        uniform_stride_cfg=uniform_stride_cfg,
    )
    assert uniform_stride >= 1, "uniform_stride must be >= 1"

    z_idx = np.arange(0, z_dim, uniform_stride, dtype=np.int64)
    y_idx = np.arange(0, y_dim, uniform_stride, dtype=np.int64)
    x_idx = np.arange(0, x_dim, uniform_stride, dtype=np.int64)
    zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
    candidate = np.ravel_multi_index((zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)), dims=alpha.shape)
    kept = candidate[flat_alpha[candidate] > alpha_threshold]
    assert kept.size > 0, "No voxels remain after TF alpha filtering"

    target_n_points = min(int(n_points), int(kept.size))
    picked = np.random.choice(int(kept.size), size=target_n_points, replace=False)
    return kept[picked]


def sample_indices_with_alpha_weights(alpha: np.ndarray, n_points: int) -> np.ndarray:
    flat_alpha = np.clip(np.asarray(alpha, dtype=np.float32).reshape(-1), 0.0, None)
    probs = flat_alpha
    if probs.sum() <= 1e-12:
        probs = np.ones_like(probs, dtype=np.float32)
    probs = probs / probs.sum()

    nonzero_indices = np.flatnonzero(probs > 0)
    nonzero = int(nonzero_indices.shape[0])
    assert nonzero > 0, "No nonzero-probability voxels available for sampling"

    target_n_points = min(int(n_points), nonzero)
    weights = probs[nonzero_indices]
    weights = weights / weights.sum()
    picked = np.random.choice(nonzero, size=target_n_points, replace=False, p=weights)
    return nonzero_indices[picked]


def sample_jitter(ijk: np.ndarray, use_jitter: bool) -> np.ndarray:
    if not use_jitter:
        return ijk
    jitter = np.random.uniform(low=-0.5, high=0.5, size=ijk.shape).astype(np.float32)
    return ijk + jitter


def build_anisotropic_attributes(
    vol,
    alpha: np.ndarray,
    z_idx_i: np.ndarray,
    y_idx_i: np.ndarray,
    x_idx_i: np.ndarray,
    cfg,
) -> dict[str, np.ndarray]:
    field_name = str(cfg.get("aniso_field", "alpha")).strip().lower()
    if field_name == "alpha":
        field_zyx = alpha.astype(np.float32, copy=False)
    elif field_name == "scalar":
        if vol.data.ndim == 3:
            field_zyx = np.asarray(vol.data, dtype=np.float32)
        else:
            field_zyx = np.asarray(vol.data[0], dtype=np.float32)
    else:
        raise ValueError(f"aniso_field must be 'alpha' or 'scalar', got {field_name!r}")

    grad_zyx = scalar_gradient_index_to_world(vol, field_zyx)
    log_scales, rots, normals_n = anisotropic_scales_and_rots(
        vol,
        grad_zyx,
        z_idx_i,
        y_idx_i,
        x_idx_i,
        float(cfg.get("aniso_tangent_scale_mul", 1.5)),
        float(cfg.get("aniso_normal_scale_mul", 0.1)),
        str(cfg.get("aniso_voxel_size_mode", "mean_spacing")),
        float(cfg.get("aniso_grad_eps", 1e-6)),
        float(cfg.get("aniso_min_linear_scale", 1e-6)),
    )

    tr = resolve_render_world_transform(cfg)
    if tr is not None and "scale_factor" in tr:
        apply_uniform_scale_to_log_scales(log_scales, float(tr["scale_factor"]))

    return {
        "scale_0": log_scales[:, 0:1],
        "scale_1": log_scales[:, 1:2],
        "scale_2": log_scales[:, 2:3],
        "rot_0": rots[:, 0:1],
        "rot_1": rots[:, 1:2],
        "rot_2": rots[:, 2:3],
        "rot_3": rots[:, 3:4],
        "nx": normals_n[:, 0:1],
        "ny": normals_n[:, 1:2],
        "nz": normals_n[:, 2:3],
    }


def _resolve_uniform_stride(
    alpha_shape: tuple[int, int, int],
    total: int,
    n_points: int,
    flat_alpha: np.ndarray,
    alpha_threshold: float,
    uniform_stride_cfg,
) -> int:
    if isinstance(uniform_stride_cfg, str):
        stride_raw = uniform_stride_cfg.strip().lower()
        auto_stride = stride_raw == "auto"
        uniform_stride = 1 if auto_stride else int(stride_raw)
    else:
        uniform_stride = int(uniform_stride_cfg)
        auto_stride = uniform_stride <= 0

    if not auto_stride:
        return uniform_stride

    z_dim, y_dim, x_dim = alpha_shape

    def candidate_count_for_stride(s: int) -> int:
        return int(np.ceil(z_dim / s) * np.ceil(y_dim / s) * np.ceil(x_dim / s))

    keep_ratio = float(np.mean(flat_alpha > alpha_threshold))
    keep_ratio = max(keep_ratio, 1e-6)
    approx_candidates = float(n_points) / keep_ratio
    approx_stride = int(round((float(total) / max(approx_candidates, 1.0)) ** (1.0 / 3.0)))
    approx_stride = max(1, approx_stride)

    best_stride = approx_stride
    best_gap = float("inf")
    for stride in range(max(1, approx_stride - 4), approx_stride + 5):
        expected_kept = candidate_count_for_stride(stride) * keep_ratio
        gap = abs(expected_kept - float(n_points))
        if gap < best_gap:
            best_gap = gap
            best_stride = stride
    return best_stride
