# Expose stable core API
from .core.types import CanonicalVolumeArtifact, VolumeTile, Volume, PointCloud, Metadata
from .core.pipeline import run_pipeline, Reader, Stage, Sampler, Renderer, Writer
from .core.errors import Vol2PcError
from .config import Config, load_config
from .registry import register_builtin_plugins

__all__ = [
    "CanonicalVolumeArtifact",
    "VolumeTile",
    "Volume", "PointCloud", "Metadata",
    "run_pipeline", "Reader", "Stage", "Sampler", "Renderer", "Writer",
    "Vol2PcError",
    "Config", "load_config",
    "register_builtin_plugins"
]

# Version
__version__ = "0.1.0"
