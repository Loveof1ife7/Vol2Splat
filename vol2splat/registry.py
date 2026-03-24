from typing import Dict, Type
from .core.pipeline import Reader, Stage, Sampler, Renderer, Writer
from .core.errors import RegistryError
from .core.log import get_logger

logger = get_logger('registry')

READERS: Dict[str, Type[Reader]] = {}
STAGES: Dict[str, Type[Stage]] = {}
SAMPLERS: Dict[str, Type[Sampler]] = {}
RENDERERS: Dict[str, Type[Renderer]] = {}
WRITERS: Dict[str, Type[Writer]] = {}

def register_reader(name: str, cls: Type[Reader]) -> None:
    if name in READERS:
        logger.warning(f"Reader '{name}' is being overwritten.")
    READERS[name] = cls

def register_stage(name: str, cls: Type[Stage]) -> None:
    if name in STAGES:
        logger.warning(f"Stage '{name}' is being overwritten.")
    STAGES[name] = cls

def register_sampler(name: str, cls: Type[Sampler]) -> None:
    if name in SAMPLERS:
        logger.warning(f"Sampler '{name}' is being overwritten.")
    SAMPLERS[name] = cls

def register_renderer(name: str, cls: Type[Renderer]) -> None:
    if name in RENDERERS:
        logger.warning(f"Renderer '{name}' is being overwritten.")
    RENDERERS[name] = cls

def register_writer(name: str, cls: Type[Writer]) -> None:
    if name in WRITERS:
        logger.warning(f"Writer '{name}' is being overwritten.")
    WRITERS[name] = cls

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

def get_renderer(name: str) -> Type[Renderer]:
    if name not in RENDERERS:
        raise RegistryError(f"Renderer '{name}' not found. Available: {list(RENDERERS.keys())}")
    return RENDERERS[name]

def get_writer(name: str) -> Type[Writer]:
    if name not in WRITERS:
        raise RegistryError(f"Writer '{name}' not found. Available: {list(WRITERS.keys())}")
    return WRITERS[name]

def register_builtin_plugins():
    logger.info('Loading built-in plugins...')
    for module_name in ['io', 'preprocess', 'rendering', 'sampling', 'export']:
        try:
            __import__(f'vol2splat.{module_name}')
        except Exception as e:
            logger.error(f'Failed to load {module_name} plugins: {e}')
