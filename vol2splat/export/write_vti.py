import os
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
except Exception:
    vtk = None
    numpy_to_vtk = None

from ..core.types import Volume


def _require_vtk() -> None:
    if vtk is None or numpy_to_vtk is None:
        raise ImportError("vtk is required to write VTI files")


def _set_direction(image, direction: np.ndarray) -> None:
    if not hasattr(image, "SetDirectionMatrix"):
        return
    matrix = vtk.vtkMatrix3x3()
    for row in range(3):
        for col in range(3):
            matrix.SetElement(row, col, float(direction[row, col]))
    image.SetDirectionMatrix(matrix)


def write_volume_to_vti(vol: Volume, path: str, array_name: str = "Scalars") -> str:
    _require_vtk()
    if vol.data.ndim != 3:
        raise ValueError(f"write_volume_to_vti expects scalar 3D volume, got shape {vol.data.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    data_zyx = np.asarray(vol.data)
    data_xyz = np.transpose(data_zyx, (2, 1, 0))
    dims = tuple(int(v) for v in data_xyz.shape)
    flat = np.ascontiguousarray(data_xyz).ravel(order="F")

    vtk_type = numpy_to_vtk(num_array=np.asarray([0], dtype=data_xyz.dtype), deep=False).GetDataType()
    vtk_array = numpy_to_vtk(num_array=flat, deep=True, array_type=vtk_type)
    vtk_array.SetName(array_name)

    image = vtk.vtkImageData()
    image.SetDimensions(dims)
    image.SetSpacing(float(vol.spacing[0]), float(vol.spacing[1]), float(vol.spacing[2]))
    image.SetOrigin(float(vol.origin[0]), float(vol.origin[1]), float(vol.origin[2]))
    _set_direction(image, np.asarray(vol.direction, dtype=np.float64))
    image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image)
    writer.Write()
    return path
