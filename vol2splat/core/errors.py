class Vol2SplatError(Exception):
    """Base exception for the vol2splat library."""


class Vol2PcError(Vol2SplatError):
    """Backward-compatible alias for older imports."""
    pass


class DataIOError(Vol2SplatError):
    """Raised when IO operations fail."""
    pass


class ConfigError(Vol2SplatError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(Vol2SplatError):
    """Raised when preprocessing fails."""
    pass


class SamplingError(Vol2SplatError):
    """Raised when sampling fails."""
    pass


class ExportError(Vol2SplatError):
    """Raised when export fails."""
    pass


class RegistryError(Vol2SplatError):
    """Raised when registry lookup fails."""
    pass
