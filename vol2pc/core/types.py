import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

@dataclass
class Metadata:
    """Metadata for volume or point cloud."""
    source_path: Optional[str] = None
    original_dtype: Optional[str] = None
    units: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

class Volume:
    """
    Represents a volumetric dataset with spatial information.
    Coordinate system: Index space (i, j, k) -> World space (x, y, z)
    """
    def __init__(
        self,
        data: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Optional[np.ndarray] = None,
        metadata: Optional[Metadata] = None
    ):
        self.data = data
        self.spacing = np.array(spacing, dtype=np.float64)
        self.origin = np.array(origin, dtype=np.float64)
        
        if direction is None:
            self.direction = np.eye(3, dtype=np.float64)
        else:
            self.direction = np.array(direction, dtype=np.float64)
            
        self.metadata = metadata or Metadata()
        self.cache: Dict[str, Any] = {}

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """
        Convert index coordinates to world coordinates.
        Args:
            ijk: (N, 3) array of index coordinates (i, j, k) -> (z, y, x) usually, 
                 but standard usually implies (x, y, z) for spatial. 
                 Let's stick to user convention or standard? 
                 User said: Volume.data : np.ndarray, shape (Z,Y,X)
                 User said: Volume.index_to_world(i,j,k) -> xyz
                 Usually in medical imaging:
                 i -> X index
                 j -> Y index
                 k -> Z index
                 But data is (Z, Y, X) => (k, j, i) access.
                 Let's assume input 'ijk' corresponds to (x_idx, y_idx, z_idx).
                 
                 World = Origin + Direction * (Index * Spacing)
        """
        # Ensure input is float for calculation
        ijk = np.asarray(ijk, dtype=np.float64)
        
        # Apply spacing
        # If ijk is (N, 3), and spacing is (3,), we multiply element-wise
        scaled_indices = ijk * self.spacing
        
        # Apply direction matrix (3x3)
        # rotated = direction @ scaled_indices.T
        # But for N points: rotated = scaled_indices @ direction.T
        rotated = scaled_indices @ self.direction.T
        
        # Add origin
        world_coords = rotated + self.origin
        return world_coords

    def world_to_index(self, xyz: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to index coordinates.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        
        # Subtract origin
        centered = xyz - self.origin
        
        # Inverse rotation
        try:
            inv_direction = np.linalg.inv(self.direction)
        except np.linalg.LinAlgError:
            raise ValueError("Direction matrix is singular")
            
        unrotated = centered @ inv_direction.T
        
        # Remove spacing
        indices = unrotated / self.spacing
        return indices

@dataclass
class PointCloud:
    """
    Represents a point cloud with attributes.
    """
    xyz: np.ndarray  # (N, 3) float32
    attrs: Dict[str, np.ndarray] = field(default_factory=dict) # key -> (N, *)
    metadata: Optional[Metadata] = None

    def __post_init__(self):
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
