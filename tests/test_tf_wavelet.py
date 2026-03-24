import numpy as np
import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat.core.types import Volume
from vol2splat.preprocess.tf import TransferFunctionStage
from vol2splat.sampling.wavelet import WaveletSampler

class TestTFAndWavelet(unittest.TestCase):
    def test_tf_gaussian(self):
        print("\n--- Testing TF Gaussian ---")
        # Create (Z, Y, X) density volume
        D, H, W = 16, 16, 16
        data = np.linspace(0, 1, D*H*W).reshape(D, H, W).astype(np.float32)
        vol = Volume(data=data)
        
        # Run TF stage
        stage = TransferFunctionStage()
        cfg = {"device": "cpu"} # Use CPU for test simplicity
        vol_out = stage.run(vol, cfg)
        
        # Check output
        print(f"TF Output shape: {vol_out.shape}")
        self.assertEqual(vol_out.shape, (4, D, H, W))
        
        # Check values
        # Center is -0.9, width 0.2. Input 0..1.
        # Most values should be near 0 opacity because input > center significantly?
        # Gaussian: exp(-0.5 * ((s - mu) / sigma)^2)
        # s=0: exp(-0.5 * ((0 - (-0.9))/0.2)^2) = exp(-0.5 * (4.5)^2) ~ 0
        # So opacity should be low.
        pass

    def test_wavelet_multichannel(self):
        print("\n--- Testing Wavelet Multi-Channel ---")
        # Create (4, Z, Y, X) volume
        D, H, W = 32, 32, 32
        data = np.random.rand(4, D, H, W).astype(np.float32)
        vol = Volume(data=data, spacing=(1.0, 1.0, 1.0))
        
        sampler = WaveletSampler()
        cfg = {
            "n_points": 100,
            "wavelet": "haar",
            "level": 2,
            "device": "cpu"
        }
        
        pc = sampler.sample(vol, cfg)
        
        print(f"Sampled points: {pc.xyz.shape}")
        self.assertEqual(pc.xyz.shape[1], 3)
        self.assertTrue(pc.xyz.shape[0] <= 100)
        
        print("Attributes:", pc.attrs.keys())
        self.assertIn("coeff_abs", pc.attrs)

if __name__ == "__main__":
    unittest.main()
