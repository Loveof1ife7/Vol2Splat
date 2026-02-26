import sys
import os
import unittest
import numpy as np
import torch

# Ensure package import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2pc.sampling.wavelet import WaveletSampler
from vol2pc.core.types import Volume

class TestWaveletSampler(unittest.TestCase):
    def test_sample_shapes_and_attrs(self):
        Z, Y, X = 64, 64, 64
        z = np.linspace(0, 1, Z)
        y = np.linspace(0, 1, Y)
        x = np.linspace(0, 1, X)
        Zg, Yg, Xg = np.meshgrid(z, y, x, indexing='ij')
        d = np.sqrt((Xg - 0.5) ** 2 + (Yg - 0.5) ** 2 + (Zg - 0.5) ** 2)
        vol_np = np.exp(-16 * d ** 2).astype(np.float32)
        vol = Volume(data=vol_np, spacing=(0.01, 0.01, 0.01))

        sampler = WaveletSampler()
        n_points = 3000
        cfg = {
            "n_points": n_points,
            "wavelet": "haar",
            "level": 2,
            "mode": "periodic",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        pc = sampler.sample(vol, cfg)

        self.assertEqual(pc.xyz.shape[0], n_points)
        self.assertIn("coeff_abs", pc.attrs)
        self.assertIn("level", pc.attrs)
        self.assertIn("band", pc.attrs)

        # Print key info for manual inspection
        print("points:", pc.xyz.shape)
        print("coeff_abs min/max:", float(pc.attrs["coeff_abs"].min()), float(pc.attrs["coeff_abs"].max()))
        print("levels unique:", np.unique(pc.attrs["level"]).tolist())
        print("bands sample:", pc.attrs["band"][:10].tolist())

if __name__ == "__main__":
    unittest.main()
