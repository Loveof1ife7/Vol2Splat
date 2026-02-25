# Explicitly export classes for external use and to trigger registration
from .read_vti import VTIReader
# from .read_raw import RawReader # Not implemented yet but placeholder

__all__ = ["VTIReader"]
