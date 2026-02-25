from typing import Dict, Type, Optional, Any
from .core.pipeline import Reader, Stage, Sampler, Writer
from .core.errors import RegistryError
from .core.log import get_logger

logger = get_logger("registry")

READERS: Dict[str, Type[Reader]] = {}
STAGES: Dict[str, Type[Stage]] = {}
SAMPLERS: Dict[str, Type[Sampler]] = {}
WRITERS: Dict[str, Type[Writer]] = {}

def register_reader(name: str, cls: Type[Reader]) -> None:
    if name in READERS:
        logger.warning(f"Reader '{name}' is being overwritten.")
    READERS[name] = cls
    logger.debug(f"Registered Reader: {name}")

def register_stage(name: str, cls: Type[Stage]) -> None:
    if name in STAGES:
        logger.warning(f"Stage '{name}' is being overwritten.")
    STAGES[name] = cls
    logger.debug(f"Registered Stage: {name}")

def register_sampler(name: str, cls: Type[Sampler]) -> None:
    if name in SAMPLERS:
        logger.warning(f"Sampler '{name}' is being overwritten.")
    SAMPLERS[name] = cls
    logger.debug(f"Registered Sampler: {name}")

def register_writer(name: str, cls: Type[Writer]) -> None:
    if name in WRITERS:
        logger.warning(f"Writer '{name}' is being overwritten.")
    WRITERS[name] = cls
    logger.debug(f"Registered Writer: {name}")

def get_reader(name: str) -> Type[Reader]:
    if name not in READERS:
        raise RegistryError(f"Reader '{name}' not found. Available: {list(READERS.keys())}")
    return READERS[name]

def get_stage(name: str) -> Type[Stage]:
    if name not in STAGES:
        raise RegistryError(f"Stage '{name}' not found. Available: {list(STAGES.keys())}")
    return STAGES[name]

def get_sampler(name: str) -> Type[Sampler]:
    if name not in SAMPLERS:
        raise RegistryError(f"Sampler '{name}' not found. Available: {list(SAMPLERS.keys())}")
    return SAMPLERS[name]

def get_writer(name: str) -> Type[Writer]:
    if name not in WRITERS:
        raise RegistryError(f"Writer '{name}' not found. Available: {list(WRITERS.keys())}")
    return WRITERS[name]

def register_builtin_plugins():
    """Import subpackages to trigger self-registration of built-in plugins."""
    # We use local imports to avoid circular dependency if this function is called early
    # But subpackages import core.pipeline which imports registry...
    # Wait, if registry is imported by core.pipeline, importing registry is fine.
    # But if we import subpackages here, they will import core.pipeline.
    
    # Structure:
    # vol2pc.io -> vol2pc.core.pipeline -> vol2pc.registry
    # vol2pc.registry -> vol2pc.io (inside this function)
    
    # This is safe as long as we don't call register_builtin_plugins() at module level of registry.py
    
    logger.info("Loading built-in plugins...")
    
    # IO
    try:
        from . import io
        # Force import of modules inside io if they are not in io.__init__
        # But we will put them in io.__init__
    except ImportError as e:
        logger.error(f"Failed to load IO plugins: {e}")

    # Preprocess
    try:
        from . import preprocess
    except ImportError as e:
        logger.error(f"Failed to load Preprocess plugins: {e}")

    # Sampling
    try:
        from . import sampling
    except ImportError as e:
        logger.error(f"Failed to load Sampling plugins: {e}")

    # Export
    try:
        from . import export
    except ImportError as e:
        logger.error(f"Failed to load Export plugins: {e}")
