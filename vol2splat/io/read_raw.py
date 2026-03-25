import os
import numpy as np

from ..core.pipeline import Reader
from ..core.types import Metadata, Volume
from .tiling import maybe_tile_input_volume
from ..registry import register_reader


def _parse_direction(value):
    if value is None:
        return np.eye(3, dtype=np.float64)
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape == (9,):
        direction = direction.reshape(3, 3)
    if direction.shape != (3, 3):
        raise ValueError(f"direction must be shape (3, 3), got {direction.shape}")
    return direction


class RawReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        shape = kwargs.get("shape") or kwargs.get("shape_zyx")
        if shape is None:
            raise ValueError("RAW reader requires 'shape' or 'shape_zyx'")
        if len(shape) != 3:
            raise ValueError(f"RAW shape must have length 3, got {shape}")

        dtype = np.dtype(kwargs.get("dtype", np.uint16))
        endian = kwargs.get("endian", "native")
        order = kwargs.get("order", "zyx").lower()
        offset = int(kwargs.get("offset", 0))
        spacing = tuple(float(v) for v in kwargs.get("spacing", (1.0, 1.0, 1.0)))
        origin = tuple(float(v) for v in kwargs.get("origin", (0.0, 0.0, 0.0)))
        direction = _parse_direction(kwargs.get("direction"))

        if endian in {"little", "le"}:
            dtype = dtype.newbyteorder("<")
        elif endian in {"big", "be"}:
            dtype = dtype.newbyteorder(">")

        shape = tuple(int(v) for v in shape)
        count = int(np.prod(shape))
        with open(path, "rb") as f:
            if offset:
                f.seek(offset)
            data = np.fromfile(f, dtype=dtype, count=count)
        if data.size != count:
            raise ValueError(f"RAW file size does not match expected voxel count: expected {count}, got {data.size}")

        if order == "zyx":
            data_zyx = data.reshape(shape)
        elif order == "xyz":
            # If the user provides RAW shape/order in XYZ, convert to the project-wide ZYX numpy contract.
            data_xyz = data.reshape(shape)
            data_zyx = np.transpose(data_xyz, (2, 1, 0))
        else:
            raise ValueError(f"Unsupported RAW order: {order}")

        metadata = Metadata(
            source_path=os.path.abspath(path),
            original_dtype=str(data_zyx.dtype),
            extra={
                "reader": "raw",
                "shape_zyx": list(data_zyx.shape),
                "dtype": str(dtype),
                "endian": endian,
                "order": order,
                "axis_convention": {
                    "data": "zyx",
                    "spacing_origin": "xyz",
                },
            },
        )
        vol = Volume(
            data=data_zyx,
            spacing=spacing,
            origin=origin,
            direction=direction,
            metadata=metadata,
        )
        return maybe_tile_input_volume(vol, kwargs)


register_reader("raw", RawReader)
