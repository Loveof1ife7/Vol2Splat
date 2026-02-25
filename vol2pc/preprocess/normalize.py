import numpy as np
from typing import Dict, Any
from ..core.types import Volume
from ..core.pipeline import Stage
from ..registry import register_stage

class NormalizeStage(Stage):
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        method = cfg.get("method", "minmax")
        if method == "minmax":
            vmin = vol.data.min()
            vmax = vol.data.max()
            if vmax > vmin:
                vol.data = (vol.data - vmin) / (vmax - vmin)
        elif method == "percentile":
            pmin = cfg.get("pmin", 1)
            pmax = cfg.get("pmax", 99)
            vmin, vmax = np.percentile(vol.data, [pmin, pmax])
            if vmax > vmin:
                vol.data = np.clip((vol.data - vmin) / (vmax - vmin), 0, 1)
            else:
                vol.data = np.zeros_like(vol.data)
        return vol

register_stage("normalize", NormalizeStage)
