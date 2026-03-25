import os
import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except Exception:
    vtk = None
    vtk_to_numpy = None

from ..core.pipeline import Reader
from ..core.types import Metadata, Volume
from .tiling import maybe_tile_input_volume
from ..registry import register_reader


def _affine_to_spacing_origin_direction(affine: np.ndarray):
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    spacing = np.where(spacing <= 1e-12, 1.0, spacing)
    direction = axes / spacing.reshape(1, 3)
    origin = affine[:3, 3]
    return spacing.astype(np.float64), origin.astype(np.float64), direction.astype(np.float64)


class NiftiReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        if nib is not None:
            img = nib.load(path)
            # NIfTI arrays are interpreted as XYZ here, then converted to the project-wide ZYX numpy contract.
            data_xyz = np.asarray(img.get_fdata(dtype=np.float32))
            if data_xyz.ndim != 3:
                raise ValueError(f"NIfTI reader expects 3D scalar volume, got shape {data_xyz.shape}")
            data_zyx = np.transpose(data_xyz, (2, 1, 0))
            # spacing/origin/direction remain in XYZ order even though the numpy array is ZYX.
            spacing, origin, direction = _affine_to_spacing_origin_direction(img.affine)
            metadata = Metadata(
                source_path=os.path.abspath(path),
                original_dtype=str(img.get_data_dtype()),
                extra={
                    "reader": "nii",
                    "shape_xyz": list(data_xyz.shape),
                    "shape_zyx": list(data_zyx.shape),
                    "axis_convention": {
                        "data": "zyx",
                        "spacing_origin": "xyz",
                    },
                    "affine": np.asarray(img.affine, dtype=np.float64).tolist(),
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

        if vtk is None or vtk_to_numpy is None:
            raise ImportError("nibabel or vtk is required to read NIfTI files")

        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(path)
        reader.Update()
        image = reader.GetOutput()
        if image is None:
            raise ValueError(f"Failed to read NIfTI file: {path}")

        dims = tuple(int(v) for v in image.GetDimensions())
        scalars = image.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("No scalar array found in NIfTI file")
        flat = vtk_to_numpy(scalars)
        data_xyz = flat.reshape(dims, order="F")
        data_zyx = np.transpose(data_xyz, (2, 1, 0)).astype(np.float32, copy=False)
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        origin = np.asarray(image.GetOrigin(), dtype=np.float64)
        direction = np.eye(3, dtype=np.float64)
        metadata = Metadata(
            source_path=os.path.abspath(path),
            original_dtype=str(data_zyx.dtype),
            extra={
                "reader": "nii",
                "shape_xyz": list(dims),
                "shape_zyx": list(data_zyx.shape),
                "axis_convention": {
                    "data": "zyx",
                    "spacing_origin": "xyz",
                },
                "backend": "vtk",
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

register_reader("nii", NiftiReader)
register_reader("nii.gz", NiftiReader)
