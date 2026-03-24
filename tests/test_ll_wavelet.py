import torch
import sys
import os
import unittest

# Ensure vol2splat is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat.sampling.low_level.ll_wavelet import dwt3

class TestLLWavelet(unittest.TestCase):
    def test_dwt3_structure(self):
        print("Testing dwt3 output structure and shapes...")
        
        # Create a random 4D tensor (C, D, H, W)
        # Using small size for quick testing
        D, H, W = 32, 32, 32
        data = torch.randn(1, D, H, W)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # 1. Test Single Channel
        print("\n--- Single Channel Test ---")
        level = 2
        coeffs = dwt3(data, wavelet='haar', level=level, mode='periodic', device=device)
        
        # Expected structure: [Approximation, Details_Level_2, Details_Level_1]
        # Details are dicts
        print(f"Input shape: {data.shape}")
        print(f"Output list length: {len(coeffs)}")
        
        self.assertEqual(len(coeffs), level + 1)
        
        # Check Approximation (LL)
        approx = coeffs[0]
        print(f"Approximation (Level {level}) shape: {approx.shape}")
        self.assertEqual(approx.shape, (1, 8, 8, 8))
        
        # Check Details
        for i, detail in enumerate(coeffs[1:]):
            lvl = level - i
            print(f"Details Level {lvl} keys: {list(detail.keys())}")
            # Print shape of one detail coefficient
            first_key = list(detail.keys())[0]
            print(f"Details Level {lvl} ['{first_key}'] shape: {detail[first_key].shape}")
            
            expected_dim = D // (2**(level - i))
            self.assertEqual(detail[first_key].shape, (1, expected_dim, expected_dim, expected_dim))

        # 2. Test Multi Channel
        print("\n--- Multi Channel Test ---")
        data_rgba = torch.randn(4, D, H, W)
        coeffs_rgba = dwt3(data_rgba, wavelet='haar', level=level, mode='periodic', multi_channel=True, device=device)
        
        print(f"Output keys (channels): {list(coeffs_rgba.keys())}")
        self.assertEqual(set(coeffs_rgba.keys()), {'r', 'g', 'b', 'a'})
        
        # Check 'r' channel
        r_coeffs = coeffs_rgba['r']
        print(f"Channel 'r' list length: {len(r_coeffs)}")
        print(f"Channel 'r' Approx shape: {r_coeffs[0].shape}")
        self.assertEqual(len(r_coeffs), level + 1)
        self.assertEqual(r_coeffs[0].shape, (1, 8, 8, 8))

if __name__ == "__main__":
    unittest.main()
