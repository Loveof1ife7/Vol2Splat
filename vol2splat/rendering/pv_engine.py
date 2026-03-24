import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Dict, List
from ..common.spatial import compute_uniform_target_bbox_transform, compute_volume_bounds
from ..core.pipeline import Renderer
from ..io.read_vti import VTIReader
from ..registry import register_renderer

class PVEngineRenderer(Renderer):
    def render(self, canonical_vti_path: str, path: str = None, **kwargs) -> Dict[str, object]:
        canonical_vti_path = os.path.abspath(canonical_vti_path)
        if not os.path.exists(canonical_vti_path):
            raise FileNotFoundError(f'Canonical VTI not found: {canonical_vti_path}')
        out_dir = os.path.abspath(path or kwargs.pop('out', None) or kwargs.pop('out_dir', None) or '.')
        os.makedirs(out_dir, exist_ok=True)
        executable = kwargs.pop('executable', kwargs.pop('pvpython', 'pvpython'))
        runner_script = kwargs.pop('script', self._default_runner_script())
        cmd = [executable, runner_script, '--vti', canonical_vti_path, '--out', out_dir]
        cmd.extend(self._kwargs_to_cli_flags(kwargs))
        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parents[2])
        existing_pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = repo_root if not existing_pythonpath else f'{repo_root}:{existing_pythonpath}'
        subprocess.run(cmd, check=True, env=env, cwd=repo_root)
        tf_configs = self._discover_tf_configs(out_dir)
        render_world_transform = self._compute_render_world_transform(canonical_vti_path, float(kwargs.get('scene_bbox_size', 2.6)))
        return {
            'output_dir': out_dir,
            'canonical_vti_path': canonical_vti_path,
            'tf_configs': tf_configs,
            'render_world_transform': render_world_transform,
            'command': cmd,
            'runner_script': runner_script,
        }

    def _default_runner_script(self) -> str:
        return str(Path(__file__).resolve().parent / 'engine' / 'run_exporter.py')

    def _kwargs_to_cli_flags(self, kwargs: Dict[str, object]) -> List[str]:
        flags: List[str] = []
        for key, value in kwargs.items():
            if value is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    flags.append(flag)
                continue
            if isinstance(value, (list, tuple)):
                flags.extend([flag, ','.join(str(v) for v in value)])
                continue
            flags.extend([flag, str(value)])
        return flags

    def _discover_tf_configs(self, out_dir: str) -> List[str]:
        direct = os.path.join(out_dir, 'tf_config.json')
        tf_configs: List[str] = []
        if os.path.exists(direct):
            tf_configs.append(os.path.abspath(direct))
        tf_configs.extend(sorted(os.path.abspath(path) for path in glob(os.path.join(out_dir, 'TF*', 'tf_config.json'))))
        return tf_configs

    def _compute_render_world_transform(self, canonical_vti_path: str, scene_bbox_size: float):
        try:
            with open(canonical_vti_path, 'rb') as f:
                header = f.read(64)
            if b'<VTKFile' not in header and b'<?xml' not in header:
                return None
            vol = VTIReader().read(canonical_vti_path)
            bounds = compute_volume_bounds(vol)
            return compute_uniform_target_bbox_transform(bounds, scene_bbox_size)
        except Exception:
            return None

register_renderer('pv_engine', PVEngineRenderer)
