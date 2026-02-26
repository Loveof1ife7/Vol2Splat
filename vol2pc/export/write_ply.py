import numpy as np
from typing import Dict, Any
from ..core.types import PointCloud
from ..core.pipeline import Writer
from ..registry import register_writer

class PLYWriter(Writer):
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        N = pc.xyz.shape[0]
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            attr_keys = []
            for k in sorted(list(pc.attrs.keys())):
                arr = pc.attrs[k]
                if not isinstance(arr, np.ndarray):
                    continue
                if not np.issubdtype(arr.dtype, np.number):
                    continue
                attr_keys.append(k)
                if arr.ndim > 1:
                    if arr.shape[1] == 1:
                        f.write(f"property float {k}\n")
                    elif arr.shape[1] == 3:
                        f.write(f"property float {k}_x\n")
                        f.write(f"property float {k}_y\n")
                        f.write(f"property float {k}_z\n")
                else:
                    f.write(f"property float {k}\n")
            
            f.write("end_header\n")
            
            data_list = [pc.xyz]
            for k in attr_keys:
                arr = pc.attrs[k]
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                data_list.append(arr.astype(np.float32, copy=False))
            
            data = np.hstack(data_list)
            np.savetxt(f, data, fmt="%.6f")

register_writer("ply", PLYWriter)
