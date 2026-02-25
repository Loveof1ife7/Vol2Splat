from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
import json
from .core.errors import ConfigError

@dataclass
class IOConfig:
    reader: str
    path: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StageConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SamplingConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExportConfig:
    writer: str
    path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    io: IOConfig
    preprocess: List[StageConfig] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(name="uniform"))
    export: Optional[ExportConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        try:
            # IO
            io_data = data.get('io', {})
            if not io_data:
                raise ConfigError("Missing 'io' section in config")
            
            io_reader = io_data.get('reader')
            if not io_reader:
                raise ConfigError("Missing 'io.reader'")
            
            io_path = io_data.get('path')
            io_raw = {k: v for k, v in io_data.items() if k not in ['reader', 'path']}
            io_cfg = IOConfig(reader=io_reader, path=io_path, raw=io_raw)

            # Preprocess
            preprocess_data = data.get('preprocess', [])
            preprocess_cfgs = []
            for item in preprocess_data:
                if 'name' not in item:
                    raise ConfigError("Preprocess stage missing 'name'")
                name = item['name']
                params = {k: v for k, v in item.items() if k != 'name'}
                preprocess_cfgs.append(StageConfig(name=name, params=params))

            # Sampling
            sampling_data = data.get('sampling', {})
            if not sampling_data:
                 raise ConfigError("Missing 'sampling' section")
            
            samp_name = sampling_data.get('name')
            if not samp_name:
                raise ConfigError("Missing 'sampling.name'")
            samp_params = {k: v for k, v in sampling_data.items() if k != 'name'}
            sampling_cfg = SamplingConfig(name=samp_name, params=samp_params)

            # Export
            export_cfg = None
            export_data = data.get('export')
            if export_data:
                writer_name = export_data.get('writer')
                if not writer_name:
                    raise ConfigError("Missing 'export.writer'")
                export_path = export_data.get('path')
                export_params = {k: v for k, v in export_data.items() if k not in ['writer', 'path']}
                export_cfg = ExportConfig(writer=writer_name, path=export_path, params=export_params)

            return cls(
                io=io_cfg,
                preprocess=preprocess_cfgs,
                sampling=sampling_cfg,
                export=export_cfg
            )
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f"Failed to parse config: {str(e)}")

def load_config(path: str) -> Config:
    try:
        with open(path, 'r') as f:
            if path.endswith('.json'):
                data = json.load(f)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(f)
        
        return Config.from_dict(data)
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parse error: {str(e)}")
    except json.JSONDecodeError as e:
        raise ConfigError(f"JSON parse error: {str(e)}")
