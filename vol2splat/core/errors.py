class Vol2PcError(Exception):
    """Base exception for vol2splat library."""
    pass

class DataIOError(Vol2PcError):
    """Raised when IO operations fail."""
    pass

class ConfigError(Vol2PcError):
    """Raised when configuration is invalid."""
    pass

class ProcessingError(Vol2PcError):
    """Raised when preprocessing fails."""
    pass

class SamplingError(Vol2PcError):
    """Raised when sampling fails."""
    pass

class ExportError(Vol2PcError):
    """Raised when export fails."""
    pass

class RegistryError(Vol2PcError):
    """Raised when registry lookup fails."""
    pass
