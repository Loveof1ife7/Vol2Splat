from typing import Any, Dict

from ..common.tf import bake_rgba_volume
from ..core.pipeline import Stage
from ..core.types import Volume
from ..registry import register_stage


class TransferFunctionStage(Stage):
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        tf_json = cfg.get("tf_json")
        backend = cfg.get("backend", cfg.get("tf_backend", "auto"))
        device = cfg.get("device", cfg.get("tf_device"))
        premultiply_alpha = bool(cfg.get("premultiply_alpha", True))

        if vol.data.ndim != 3:
            raise ValueError(f"TF stage expects scalar volume (Z,Y,X), got {vol.data.shape}")

        vol.data = bake_rgba_volume(
            vol.data,
            tf_json_path=tf_json,
            premultiply_alpha=premultiply_alpha,
            backend=backend,
            device=device,
            return_numpy=True,
        )
        vol.metadata.extra["transfer_function"] = {
            "tf_json": tf_json,
            "backend": backend,
            "premultiply_alpha": premultiply_alpha,
        }
        return vol


register_stage("tf", TransferFunctionStage)
