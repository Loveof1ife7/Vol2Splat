import os
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except Exception:
    vtk = None
    vtk_to_numpy = None

from ..core.types import Metadata, Volume
from ..core.pipeline import Reader
from ..registry import register_reader


def _require_vtk() -> None:
    if vtk is None or vtk_to_numpy is None:
        raise ImportError("vtk is required to read VTI files")


def _extract_direction(image) -> np.ndarray:
    if hasattr(image, "GetDirectionMatrix"):
        matrix = image.GetDirectionMatrix()
        if matrix is not None:
            direction = np.eye(3, dtype=np.float64)
            for row in range(3):
                for col in range(3):
                    direction[row, col] = matrix.GetElement(row, col)
            return direction
    return np.eye(3, dtype=np.float64)


class VTIReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        _require_vtk()
        array_name = kwargs.get("array_name")

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()
        image = reader.GetOutput()
        if image is None:
            raise ValueError(f"Failed to read VTI file: {path}")

        dims = tuple(int(v) for v in image.GetDimensions())
        spacing = tuple(float(v) for v in image.GetSpacing())
        origin = tuple(float(v) for v in image.GetOrigin())
        direction = _extract_direction(image)

        point_data = image.GetPointData()
        scalars = point_data.GetArray(array_name) if array_name else point_data.GetScalars()
        if scalars is None and array_name is not None:
            scalars = point_data.GetScalars()
        if scalars is None:
            raise ValueError("No scalar array found in VTI file")

        flat = vtk_to_numpy(scalars)
        if flat.size != dims[0] * dims[1] * dims[2]:
            raise ValueError(f"Unexpected scalar size {flat.size} for dimensions {dims}")

        # VTK image data is treated as XYZ on read, then converted to the project-wide ZYX numpy layout.
        data_xyz = flat.reshape(dims, order="F")
        data_zyx = np.transpose(data_xyz, (2, 1, 0))

        metadata = Metadata(
            source_path=os.path.abspath(path),
            original_dtype=str(data_zyx.dtype),
            extra={
                "reader": "vti",
                "array_name": scalars.GetName() or array_name or "Scalars",
                "dims_xyz": list(dims),
                "shape_zyx": list(data_zyx.shape),
                "axis_convention": {
                    "data": "zyx",
                    "spacing_origin": "xyz",
                },
            },
        )
        return Volume(
            data=data_zyx,
            spacing=spacing,
            origin=origin,
            direction=direction,
            metadata=metadata,
        )


register_reader("vti", VTIReader)
