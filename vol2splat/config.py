from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from .core.errors import ConfigError

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

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
class RenderConfig:
    renderer: str
    path: Optional[str] = None
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
    sampling: Optional[SamplingConfig] = None
    render: Optional[RenderConfig] = None
    export: Optional[ExportConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        try:
            io_data = data.get('io', {})
            if not io_data:
                raise ConfigError("Missing 'io' section in config")
            io_reader = io_data.get('reader')
            if not io_reader:
                raise ConfigError("Missing 'io.reader'")
            io_path = io_data.get('path')
            io_raw = {k: v for k, v in io_data.items() if k not in ['reader', 'path']}
            io_cfg = IOConfig(reader=io_reader, path=io_path, raw=io_raw)

            preprocess_cfgs = []
            for item in data.get('preprocess', []):
                if 'name' not in item:
                    raise ConfigError("Preprocess stage missing 'name'")
                preprocess_cfgs.append(StageConfig(name=item['name'], params={k: v for k, v in item.items() if k != 'name'}))

            sampling_cfg = None
            sampling_data = data.get('sampling')
            if sampling_data:
                samp_name = sampling_data.get('name')
                if not samp_name:
                    raise ConfigError("Missing 'sampling.name'")
                sampling_cfg = SamplingConfig(
                    name=samp_name,
                    params={k: v for k, v in sampling_data.items() if k != 'name'},
                )

            render_cfg = None
            render_data = data.get('render')
            if render_data:
                renderer_name = render_data.get('renderer')
                if not renderer_name:
                    raise ConfigError("Missing 'render.renderer'")
                render_cfg = RenderConfig(
                    renderer=renderer_name,
                    path=render_data.get('path'),
                    params={k: v for k, v in render_data.items() if k not in ['renderer', 'path']},
                )

            export_cfg = None
            export_data = data.get('export')
            if export_data:
                writer_name = export_data.get('writer')
                if not writer_name:
                    raise ConfigError("Missing 'export.writer'")
                export_cfg = ExportConfig(
                    writer=writer_name,
                    path=export_data.get('path'),
                    params={k: v for k, v in export_data.items() if k not in ['writer', 'path']},
                )

            return cls(io=io_cfg, preprocess=preprocess_cfgs, sampling=sampling_cfg, render=render_cfg, export=export_cfg)
        except Exception as e:
            if isinstance(e, ConfigError):
                raise
            raise ConfigError(f'Failed to parse config: {e}')

def load_config_data(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                return json.load(f)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                if yaml is None:
                    raise ConfigError('PyYAML is required to load YAML config files')
                return yaml.safe_load(f)
            else:
                if yaml is None:
                    raise ConfigError('PyYAML is required to load non-JSON config files')
                return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f'Config file not found: {path}')
    except json.JSONDecodeError as e:
        raise ConfigError(f'JSON parse error: {e}')
    except Exception as e:
        if yaml is not None and isinstance(e, yaml.YAMLError):
            raise ConfigError(f'YAML parse error: {e}')
        raise


def load_config(path: str) -> Config:
    return Config.from_dict(load_config_data(path))
