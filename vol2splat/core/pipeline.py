import os
from abc import ABC, abstractmethod
from typing import Any, Dict
from .types import Volume, PointCloud


class Reader(ABC):
    @abstractmethod
    def read(self, path: str, **kwargs) -> Volume:
        pass


class Stage(ABC):
    @abstractmethod
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        pass


class Sampler(ABC):
    @abstractmethod
    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        pass

    def sample_canonical_vti(self, canonical_vti_path: str, cfg: Dict[str, Any]) -> PointCloud:
        raise NotImplementedError('This sampler does not implement canonical-VTI input')


class Renderer(ABC):
    @abstractmethod
    def render(self, canonical_vti_path: str, path: str = None, **kwargs) -> Any:
        pass


class Writer(ABC):
    @abstractmethod
    def write(self, pc: PointCloud, path: str, **kwargs) -> None:
        pass


def _default_export_filename(writer_name: str) -> str:
    return {'ply': 'points.ply', 'npz': 'points.npz'}.get(writer_name, f'points.{writer_name}')


def _resolve_tf_sampling_tasks(sampler_name: str, sampler_params: Dict[str, Any]) -> list[Dict[str, Any]]:
    explicit_selector = any(sampler_params.get(key) is not None for key in ['tf_json', 'tf_config', 'tf_config_json', 'tf_name', 'tf_index'])
    if (sampler_name != 'opacity' and not bool(sampler_params.get('sample_all_tf', False))) or explicit_selector:
        return [dict(sampler_params)]
    render_result = sampler_params.get('render_result')
    if isinstance(render_result, dict):
        tf_outputs = render_result.get('tf_outputs') or []
        if len(tf_outputs) > 1:
            tasks = []
            for item in tf_outputs:
                task = dict(sampler_params)
                task['tf_json'] = item['tf_json']
                task['tf_name'] = item['tf_name']
                task['tf_output_dir'] = item['tf_dir']
                task.pop('tf_index', None)
                tasks.append(task)
            return tasks
    try:
        from ..sampling.utils import resolve_tf_json_paths
        tf_paths = resolve_tf_json_paths(sampler_params)
    except Exception:
        return [dict(sampler_params)]
    if len(tf_paths) <= 1:
        return [dict(sampler_params)]
    tasks = []
    for tf_path in tf_paths:
        task = dict(sampler_params)
        task['tf_json'] = tf_path
        task['tf_name'] = os.path.basename(os.path.dirname(tf_path))
        task['tf_output_dir'] = os.path.dirname(tf_path)
        task.pop('tf_index', None)
        tasks.append(task)
    return tasks


def _resolve_export_path_for_task(base_path: str | None, writer_name: str, sampler_task: Dict[str, Any], render_result: Any, multi_tf_mode: bool) -> str | None:
    if not multi_tf_mode:
        return base_path
    tf_name = sampler_task.get('tf_name')
    tf_output_dir = sampler_task.get('tf_output_dir')
    if not tf_name:
        return base_path
    if tf_output_dir:
        if base_path:
            _, file_name = os.path.split(base_path)
            root, ext = os.path.splitext(file_name)
            if ext:
                return os.path.join(tf_output_dir, file_name)
            return os.path.join(tf_output_dir, _default_export_filename(writer_name))
        return os.path.join(tf_output_dir, _default_export_filename(writer_name))
    if base_path:
        parent_dir, file_name = os.path.split(base_path)
        root, ext = os.path.splitext(file_name)
        if ext:
            return os.path.join(parent_dir, tf_name, file_name)
        return os.path.join(base_path, tf_name, _default_export_filename(writer_name))
    if isinstance(render_result, dict) and render_result.get('output_dir'):
        return os.path.join(render_result['output_dir'], tf_name, _default_export_filename(writer_name))
    return None


def run_pipeline(input_path: str, output_path: str, config: Any) -> PointCloud:
    from ..registry import get_reader, get_stage, get_sampler, get_writer, get_renderer

    reader_name = config.io.reader
    reader = get_reader(reader_name)()
    path_to_read = input_path if input_path else config.io.path
    print(f'Reading volume from {path_to_read} using {reader_name}...')
    vol = reader.read(path_to_read, **config.io.raw)

    for stage_cfg in config.preprocess:
        stage_name = stage_cfg.name
        print(f'Running stage {stage_name}...')
        vol = get_stage(stage_name)().run(vol, stage_cfg.params)

    canonical_vti_path = vol.cache.get('canonical_vti_path')
    if getattr(config, 'render', None):
        if not canonical_vti_path:
            raise ValueError("Render stage requires preprocess to produce 'canonical_vti_path'")
        renderer = get_renderer(config.render.renderer)()
        print(f'Rendering using {config.render.renderer}...')
        vol.cache['render_result'] = renderer.render(canonical_vti_path, path=config.render.path, **config.render.params)

    render_result = vol.cache.get('render_result')
    sampler_name = config.sampling.name
    sampler_params = dict(config.sampling.params)
    if render_result is not None:
        sampler_params.setdefault('render_result', render_result)
        if isinstance(render_result, dict) and render_result.get('output_dir'):
            sampler_params.setdefault('render_output_dir', render_result['output_dir'])
        if isinstance(render_result, dict) and render_result.get('render_world_transform'):
            sampler_params.setdefault('render_world_transform', render_result['render_world_transform'])
    sampler = get_sampler(sampler_name)()
    sampling_tasks = _resolve_tf_sampling_tasks(sampler_name, sampler_params)
    multi_tf_mode = len(sampling_tasks) > 1

    writer = None
    writer_name = None
    writer_params = None
    base_write_path = output_path if output_path else (config.export.path if config.export else None)
    if config.export:
        writer_name = config.export.writer
        writer_params = dict(config.export.params)
        if render_result is not None:
            writer_params.setdefault('render_result', render_result)
            if isinstance(render_result, dict) and render_result.get('render_world_transform'):
                writer_params.setdefault('render_world_transform', render_result['render_world_transform'])
        writer = get_writer(writer_name)()

    last_pc = None
    for task in sampling_tasks:
        label = task.get('tf_name')
        tf_json = task.get('tf_json')
        print(f"Sampling using {sampler_name}{' (' + label + ')' if label else ''}...")
        if tf_json:
            print(f"Using transfer function: {tf_json}")
        if canonical_vti_path:
            task.setdefault('canonical_vti_path', canonical_vti_path)
            try:
                pc = sampler.sample_canonical_vti(canonical_vti_path, task)
            except NotImplementedError:
                pc = sampler.sample(vol, task)
        else:
            pc = sampler.sample(vol, task)
        last_pc = pc
        if writer is not None:
            path_to_write = _resolve_export_path_for_task(base_write_path, writer_name, task, render_result, multi_tf_mode)
            if path_to_write:
                print(f'Writing to {path_to_write} using {writer_name}...')
                writer.write(pc, path_to_write, **writer_params)
            else:
                print('No output path specified, skipping write.')
    return last_pc
