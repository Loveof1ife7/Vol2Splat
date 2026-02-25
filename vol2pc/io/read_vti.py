import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
from ..core.types import Volume, Metadata
from ..core.pipeline import Reader
from ..registry import register_reader

class VTIReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()
        image = reader.GetOutput()
        
        dims = image.GetDimensions()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        
        # Get point data
        pd = image.GetPointData()
        scalars = pd.GetScalars()
        if scalars is None:
            raise ValueError("No scalars found in VTI file")
            
        data = vtk_to_numpy(scalars)
        # Reshape: VTK is (X, Y, Z) in FORTRAN order or C order?
        # VTK flattens as x varies fastest.
        # Numpy usually (Z, Y, X).
        # So we reshape to (Z, Y, X).
        data = data.reshape(dims[2], dims[1], dims[0])
        
        return Volume(data=data, spacing=spacing, origin=origin, metadata=Metadata(source_path=path))

register_reader("vti", VTIReader)
