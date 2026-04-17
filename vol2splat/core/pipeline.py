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
    return {'ply': 'points.ply', 'gs_ply': 'points.ply', 'npz': 'points.npz'}.get(writer_name, f'points.{writer_name}')


def _normalize_export_path(base_path: str | None, writer_name: str) -> str | None:
    if not base_path:
        return None
    _, ext = os.path.splitext(base_path)
    if ext:
        return base_path
    return os.path.join(base_path, _default_export_filename(writer_name))


def _scope_path_for_tile(path: str | None, tile_name: str | None) -> str | None:
    if not path or not tile_name:
        return path
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}__{tile_name}{ext}"
    return os.path.join(path, tile_name)


def _scope_stage_params_for_tile(stage_name: str, params: Dict[str, Any], tile_name: str | None) -> Dict[str, Any]:
    scoped = dict(params)
    if not tile_name:
        return scoped
    if stage_name == 'canonicalize' and scoped.get('vti_path'):
        scoped['vti_path'] = _scope_path_for_tile(scoped['vti_path'], tile_name)
    return scoped


def _scope_qc_cfg_for_tile(qc_cfg: Dict[str, Any] | None, tile_name: str | None) -> Dict[str, Any] | None:
    if not qc_cfg:
        return qc_cfg
    scoped = dict(qc_cfg)
    if tile_name:
        if scoped.get('report_json'):
            scoped['report_json'] = _scope_path_for_tile(scoped['report_json'], tile_name)
        if scoped.get('report_md'):
            scoped['report_md'] = _scope_path_for_tile(scoped['report_md'], tile_name)
    return scoped


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
        return _normalize_export_path(base_path, writer_name)
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


def _run_render_qc_if_enabled(render_result: Any, qc_cfg: Dict[str, Any] | None) -> Any:
    if not qc_cfg:
        return None
    if not isinstance(render_result, dict):
        return None
    qc_enabled = bool(qc_cfg.get('enabled', True))
    if not qc_enabled:
        return None
    from ..rendering.qc import run_render_qc
    qc_result = run_render_qc(render_result, qc_cfg)
    render_result['qc'] = qc_result
    return qc_result


def _filter_sampling_tasks_by_qc(sampling_tasks: list[Dict[str, Any]], render_result: Any) -> list[Dict[str, Any]]:
    if not isinstance(render_result, dict):
        return sampling_tasks
    qc_result = render_result.get('qc')
    if not isinstance(qc_result, dict):
        return sampling_tasks
    if not bool(qc_result.get('skip_failed_tf', True)):
        return sampling_tasks
    passed_tf_names = set(qc_result.get('passed_tf_names') or [])
    if not passed_tf_names:
        return []
    filtered = []
    for task in sampling_tasks:
        tf_name = task.get('tf_name')
        if tf_name is None or tf_name in passed_tf_names:
            filtered.append(task)
    return filtered


def _run_single_volume_pipeline(vol: Volume, output_path: str | None, config: Any, tile_name: str | None = None) -> Dict[str, Any]:
    from ..registry import get_stage, get_sampler, get_writer, get_renderer

    for stage_cfg in config.preprocess:
        stage_name = stage_cfg.name
        stage_params = _scope_stage_params_for_tile(stage_name, stage_cfg.params, tile_name)
        print(f'Running stage {stage_name}...')
        vol = get_stage(stage_name)().run(vol, stage_params)

    canonical_vti_path = vol.cache.get('canonical_vti_path')
    render_result = None
    if getattr(config, 'render', None):
        if not canonical_vti_path:
            raise ValueError("Render stage requires preprocess to produce 'canonical_vti_path'")
        renderer = get_renderer(config.render.renderer)()
        print(f'Rendering using {config.render.renderer}...')
        render_params = dict(config.render.params)
        render_qc_cfg = _scope_qc_cfg_for_tile(render_params.pop('qc', None), tile_name)
        render_path = _scope_path_for_tile(config.render.path, tile_name)
        render_result = renderer.render(canonical_vti_path, path=render_path, **render_params)
        vol.cache['render_result'] = render_result
        _run_render_qc_if_enabled(render_result, render_qc_cfg)

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
    sampling_tasks = _filter_sampling_tasks_by_qc(sampling_tasks, render_result)
    multi_tf_mode = len(sampling_tasks) > 1

    writer = None
    writer_name = None
    writer_params = None
    export_base_path = output_path if output_path else (config.export.path if config.export else None)
    export_base_path = _scope_path_for_tile(export_base_path, tile_name)
    if config.export:
        writer_name = config.export.writer
        writer_params = dict(config.export.params)
        if render_result is not None:
            writer_params.setdefault('render_result', render_result)
            if isinstance(render_result, dict) and render_result.get('render_world_transform'):
                writer_params.setdefault('render_world_transform', render_result['render_world_transform'])
        writer = get_writer(writer_name)()

    empty_sampling_error_tokens = (
        'No voxels remain after TF alpha filtering',
        'No nonzero-probability voxels available for sampling',
    )
    last_pc = None
    written_paths = []
    sampled_tasks = []
    skipped_sampling_tasks = []
    if not sampling_tasks:
        print('No sampling tasks remain after QC filtering, skipping sampling/export.')
    for task in sampling_tasks:
        label = task.get('tf_name')
        tf_json = task.get('tf_json')
        print(f"Sampling using {sampler_name}{' (' + label + ')' if label else ''}...")
        if tf_json:
            print(f"Using transfer function: {tf_json}")
        try:
            if canonical_vti_path:
                task.setdefault('canonical_vti_path', canonical_vti_path)
                try:
                    pc = sampler.sample_canonical_vti(canonical_vti_path, task)
                except NotImplementedError:
                    pc = sampler.sample(vol, task)
            else:
                pc = sampler.sample(vol, task)
        except (AssertionError, ValueError) as e:
            msg = str(e)
            if any(token in msg for token in empty_sampling_error_tokens):
                print(f"Skipping sampling/export for {label or 'default'}: {msg}")
                skipped_sampling_tasks.append({
                    'task': dict(task),
                    'reason': msg,
                })
                continue
            raise

        sampled_tasks.append(dict(task))
        last_pc = pc
        if writer is not None:
            path_to_write = _resolve_export_path_for_task(export_base_path, writer_name, task, render_result, multi_tf_mode)
            if path_to_write:
                print(f'Writing to {path_to_write} using {writer_name}...')
                writer.write(pc, path_to_write, **writer_params)
                written_paths.append(path_to_write)
            else:
                print('No output path specified, skipping write.')
    return {
        'volume': vol,
        'canonical_vti_path': canonical_vti_path,
        'render_result': render_result,
        'sampling_tasks': sampling_tasks,
        'sampled_tasks': sampled_tasks,
        'skipped_sampling_tasks': skipped_sampling_tasks,
        'point_cloud': last_pc,
        'written_paths': written_paths,
        'tile_name': tile_name,
    }


def run_pipeline_context(input_path: str, output_path: str, config: Any) -> Dict[str, Any]:
    from ..registry import get_reader

    reader_name = config.io.reader
    reader = get_reader(reader_name)()
    path_to_read = input_path if input_path else config.io.path
    print(f'Reading volume from {path_to_read} using {reader_name}...')
    vol = reader.read(path_to_read, **config.io.raw)
    io_tiles = vol.cache.get('io_tiles') or []
    if io_tiles:
        tile_results = []
        written_paths = []
        last_pc = None
        for tile in io_tiles:
            print(f'Processing input tile {tile.name}...')
            tile_result = _run_single_volume_pipeline(tile.volume, output_path, config, tile_name=tile.name)
            tile_results.append(tile_result)
            written_paths.extend(tile_result.get('written_paths', []))
            if tile_result.get('point_cloud') is not None:
                last_pc = tile_result['point_cloud']
        return {
            'volume': vol,
            'input_tiles': io_tiles,
            'tile_results': tile_results,
            'point_cloud': last_pc,
            'written_paths': written_paths,
        }

    return _run_single_volume_pipeline(vol, output_path, config, tile_name=None)


def run_pipeline(input_path: str, output_path: str, config: Any) -> PointCloud:
    context = run_pipeline_context(input_path, output_path, config)
    return context['point_cloud']
