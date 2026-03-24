import numpy as np
from ..core.types import Volume

def compute_volume_bounds(vol: Volume) -> tuple[float, float, float, float, float, float]:
    shape = vol.data.shape
    if len(shape) == 4:
        z_dim, y_dim, x_dim = shape[1], shape[2], shape[3]
    elif len(shape) == 3:
        z_dim, y_dim, x_dim = shape
    else:
        raise ValueError(f'Unsupported volume shape for bounds: {shape}')
    corners_ijk = np.array([
        [0.0, 0.0, 0.0], [x_dim - 1.0, 0.0, 0.0], [0.0, y_dim - 1.0, 0.0], [0.0, 0.0, z_dim - 1.0],
        [x_dim - 1.0, y_dim - 1.0, 0.0], [x_dim - 1.0, 0.0, z_dim - 1.0], [0.0, y_dim - 1.0, z_dim - 1.0],
        [x_dim - 1.0, y_dim - 1.0, z_dim - 1.0],
    ], dtype=np.float64)
    corners_world = vol.index_to_world(corners_ijk)
    xyz_min = corners_world.min(axis=0)
    xyz_max = corners_world.max(axis=0)
    return (float(xyz_min[0]), float(xyz_max[0]), float(xyz_min[1]), float(xyz_max[1]), float(xyz_min[2]), float(xyz_max[2]))

def compute_uniform_target_bbox_transform(bounds: tuple[float, float, float, float, float, float], target_bbox_size: float) -> dict:
    b = np.asarray(bounds, dtype=np.float64)
    ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]], dtype=np.float64)
    center = np.array([(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0, (b[4] + b[5]) / 2.0], dtype=np.float64)
    max_extent = float(np.max(ext))
    if max_extent < 1e-6:
        max_extent = 1.0
    scale_factor = float(target_bbox_size) / max_extent
    offset = -center * scale_factor
    half_ext = 0.5 * ext * scale_factor
    render_bounds = (-half_ext[0], half_ext[0], -half_ext[1], half_ext[1], -half_ext[2], half_ext[2])
    return {
        'mode': 'uniform_actor_transform',
        'scale_factor': float(scale_factor),
        'offset': offset.astype(np.float64).tolist(),
        'original_bounds': [float(v) for v in bounds],
        'render_bounds': [float(v) for v in render_bounds],
        'target_bbox_size': float(target_bbox_size),
    }

def apply_uniform_world_transform(xyz: np.ndarray, transform: dict) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    scale_factor = float(transform['scale_factor'])
    offset = np.asarray(transform['offset'], dtype=np.float32).reshape(1, 3)
    return xyz * scale_factor + offset
