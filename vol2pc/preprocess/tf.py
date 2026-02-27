import numpy as np
import torch
from typing import Dict, Any
from .low_level.ll_tf import apply_tf_volume, make_json_tf, make_gaussian_tf
from ..core.types import Volume
from ..core.pipeline import Stage
from ..registry import register_stage

class TransferFunctionStage(Stage):
    def run(self, vol: Volume, cfg: Dict[str, Any]) -> Volume:
        tf_json = cfg.get("tf_json", None)
        device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        rf = self.apply_transfer_function(vol, tf_json, device)
        return rf
    
    def apply_transfer_function(self, vol: Volume, tf_json: str, device) -> Volume:
        '''
        Return:
            Volume with data (4, D, W, H)
        '''
        # Ensure input is (D, H, W)
        if vol.data.ndim != 3:
             raise ValueError(f"TF stage expects scalar volume (D,H,W), got {vol.data.shape}")
             
        data = torch.from_numpy(vol.data.astype(np.float32)).to(device)
        
        # load transfer function to bake radiance field
        if tf_json:
            print(f"Loading Transfer Function from JSON: {tf_json}")
            tf = make_json_tf(json_path=tf_json)
        else:
            print("[WARN] Using default Gaussian Transfer Function.")
            tf = make_gaussian_tf(center=-0.9, width=0.2)

        from torch.cuda.amp import autocast
        with autocast():
            # apply_tf_volume expects (D, H, W) tensor
            rf_tensor = apply_tf_volume(
                vol=data, # [D, H, W]
                tf_callable=tf,
                premultiply_alpha=True
            ) # [D, H, W, 4]

        # Convert back to (4, D, H, W) for standard multi-channel volume
        # Output of apply_tf_volume is (D, H, W, 4)
        rf_tensor = rf_tensor.permute(3, 0, 1, 2).contiguous() # (4, D, H, W)
        
        # Update volume data
        vol.data = rf_tensor.detach().cpu().numpy()
        return vol
    
register_stage("tf", TransferFunctionStage)
