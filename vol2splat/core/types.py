import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class Metadata:
    source_path: Optional[str] = None
    original_dtype: Optional[str] = None
    units: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CanonicalVolumeArtifact:
    path: str
    shape: Tuple[int, ...]
    spacing: Tuple[float, float, float]
    origin: Tuple[float, float, float]
    direction: np.ndarray
    metadata: Optional[Metadata] = None

class Volume:
    def __init__(self, data: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None, metadata: Optional[Metadata] = None):
        self.data = data
        self.spacing = np.array(spacing, dtype=np.float64)
        self.origin = np.array(origin, dtype=np.float64)
        self.direction = np.eye(3, dtype=np.float64) if direction is None else np.array(direction, dtype=np.float64)
        self.metadata = metadata or Metadata()
        self.cache: Dict[str, Any] = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=np.float64)
        scaled_indices = ijk * self.spacing
        rotated = scaled_indices @ self.direction.T
        return rotated + self.origin

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64)
        centered = xyz - self.origin
        inv_direction = np.linalg.inv(self.direction)
        unrotated = centered @ inv_direction.T
        return unrotated / self.spacing

@dataclass
class PointCloud:
    xyz: np.ndarray
    attrs: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Optional[Metadata] = None

    def __post_init__(self):
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f'xyz must be (N, 3), got {self.xyz.shape}')
