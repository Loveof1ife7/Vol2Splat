import json
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
        tf_outputs = self._discover_tf_outputs(out_dir)
        tf_configs = [item['tf_json'] for item in tf_outputs]
        render_world_transform = self._compute_render_world_transform(canonical_vti_path, float(kwargs.get('scene_bbox_size', 2.6)))
        return {
            'output_dir': out_dir,
            'canonical_vti_path': canonical_vti_path,
            'tf_outputs': tf_outputs,
            'tf_configs': tf_configs,
            'primary_tf_json': tf_configs[0] if len(tf_configs) == 1 else None,
            'tf_count': len(tf_configs),
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
            flag = f"--{key}"
            if isinstance(value, bool):
                if value:
                    flags.append(flag)
                continue
            if isinstance(value, (list, tuple)):
                # Colormap preset names may contain commas (for example "Black, Blue and White"),
                # so serialize list-like arguments that cross the CLI boundary as JSON.
                if key in {'cmaps', 'tf_jsons'}:
                    flags.extend([flag, json.dumps([str(v) for v in value], ensure_ascii=False)])
                else:
                    flags.extend([flag, ','.join(str(v) for v in value)])
                continue
            flags.extend([flag, str(value)])
        return flags

    def _discover_tf_outputs(self, out_dir: str) -> List[Dict[str, str]]:
        outputs: List[Dict[str, str]] = []

        direct = os.path.join(out_dir, 'tf_config.json')
        if os.path.exists(direct):
            outputs.append({
                'tf_name': os.path.basename(os.path.abspath(out_dir)),
                'tf_json': os.path.abspath(direct),
                'tf_dir': os.path.abspath(out_dir),
            })

        for path in sorted(glob(os.path.join(out_dir, '*', 'tf_config.json'))):
            tf_json = os.path.abspath(path)
            tf_dir = os.path.dirname(tf_json)
            outputs.append({
                'tf_name': os.path.basename(tf_dir),
                'tf_json': tf_json,
                'tf_dir': tf_dir,
            })
        return outputs

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
