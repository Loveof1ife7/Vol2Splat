from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .types import Volume, PointCloud
from .errors import ProcessingError, SamplingError, DataIOError

class Reader(ABC):
    """Abstract base class for volume readers."""
    @abstractmethod
    def read(self, path: str, **kwargs) -> Volume:
        pass

class Stage(ABC):
    """Abstract base class for preprocessing stages."""
    @abstractmethod
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        """
        Process the volume. Should modify vol.data or vol.cache in place or return new Volume.
        """
        pass

class Sampler(ABC):
    """Abstract base class for point samplers."""
    @abstractmethod
    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        pass

class Writer(ABC):
    """Abstract base class for point cloud writers."""
    @abstractmethod
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        pass

def run_pipeline(
    input_path: str,
    output_path: str,
    config: Any, 
) -> PointCloud:
    """
    High-level pipeline execution.
    Args:
        input_path: Path to input volume.
        output_path: Path to output point cloud.
        config: Config object (duck-typed).
    """
    from ..registry import get_reader, get_stage, get_sampler, get_writer
    
    # 1. Read
    reader_name = config.io.reader
    reader_cls = get_reader(reader_name)
    reader = reader_cls()
    
    path_to_read = input_path if input_path else config.io.path
    # Allow reading from None path if reader supports it (e.g. synthetic)
    # But usually we need a path or at least config.
    
    print(f"Reading volume from {path_to_read} using {reader_name}...")
    vol = reader.read(path_to_read, **config.io.raw)
    
    # 2. Preprocess
    for stage_cfg in config.preprocess:
        stage_name = stage_cfg.name
        stage_params = stage_cfg.params
        print(f"Running stage {stage_name}...")
        stage_cls = get_stage(stage_name)
        stage = stage_cls()
        vol = stage.run(vol, stage_params)
        
    # 3. Sample
    sampler_name = config.sampling.name
    sampler_params = config.sampling.params
    print(f"Sampling using {sampler_name}...")
    sampler_cls = get_sampler(sampler_name)
    sampler = sampler_cls()
    pc = sampler.sample(vol, sampler_params)
    
    # 4. Export
    if config.export:
        writer_name = config.export.writer
        writer_params = config.export.params
        writer_cls = get_writer(writer_name)
        writer = writer_cls()
        
        path_to_write = output_path if output_path else config.export.path
        if path_to_write:
            print(f"Writing to {path_to_write} using {writer_name}...")
            writer.write(pc, path_to_write, **writer_params)
        else:
            print("No output path specified, skipping write.")
            
    return pc
