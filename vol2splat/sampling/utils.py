import os
from glob import glob
from typing import Dict, List

from ..common.spatial import apply_uniform_world_transform

try:
    import torch
except Exception:
    torch = None


def to_torch(x, dtype=None, device=None):
    if torch is None:
        raise ImportError("torch is required for to_torch")
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def discover_tf_json_paths(render_output_dir: str) -> List[str]:
    render_output_dir = os.path.abspath(render_output_dir)
    direct = os.path.join(render_output_dir, "tf_config.json")
    paths = []
    if os.path.exists(direct):
        paths.append(direct)
    paths.extend(sorted(glob(os.path.join(render_output_dir, "TF*", "tf_config.json"))))
    return [os.path.abspath(path) for path in paths]


def resolve_tf_json_paths(cfg: Dict) -> List[str]:
    explicit = cfg.get("tf_json") or cfg.get("tf_config") or cfg.get("tf_config_json")
    if explicit:
        return [os.path.abspath(explicit)]
    render_result = cfg.get("render_result")
    render_output_dir = cfg.get("render_output_dir")
    if render_output_dir:
        return discover_tf_json_paths(render_output_dir)
    if isinstance(render_result, dict):
        tf_outputs = render_result.get("tf_outputs")
        if tf_outputs:
            return [os.path.abspath(item["tf_json"]) for item in tf_outputs]
        tf_configs = render_result.get("tf_configs")
        if tf_configs:
            return [os.path.abspath(path) for path in tf_configs]
        output_dir = render_result.get("output_dir")
        if output_dir:
            return discover_tf_json_paths(output_dir)
    return []


def resolve_tf_json_path(cfg: Dict) -> str:
    paths = resolve_tf_json_paths(cfg)
    if not paths:
        raise FileNotFoundError("No transfer function JSON found for sampling")
    tf_name = cfg.get("tf_name")
    if tf_name:
        for path in paths:
            if os.path.basename(os.path.dirname(path)) == tf_name:
                return path
        raise ValueError(f"Transfer function '{tf_name}' not found. Candidates: {', '.join(os.path.basename(os.path.dirname(p)) for p in paths)}")
    tf_index = cfg.get("tf_index")
    if tf_index is not None:
        idx = int(tf_index)
        if idx < 0 or idx >= len(paths):
            raise IndexError(f"tf_index {idx} out of range for {len(paths)} transfer functions")
        return paths[idx]
    if len(paths) > 1:
        candidates = ", ".join(os.path.basename(os.path.dirname(path)) for path in paths)
        raise ValueError(f"Multiple transfer functions found. Please specify 'tf_name' or 'tf_index'. Candidates: {candidates}")
    return paths[0]

def resolve_render_world_transform(cfg: Dict):
    direct = cfg.get("render_world_transform")
    if direct:
        return direct
    render_result = cfg.get("render_result")
    if isinstance(render_result, dict):
        return render_result.get("render_world_transform")
    return None


def maybe_to_render_world(xyz, cfg: Dict):
    transform = resolve_render_world_transform(cfg)
    if not transform:
        return xyz, False
    return apply_uniform_world_transform(xyz, transform), True
