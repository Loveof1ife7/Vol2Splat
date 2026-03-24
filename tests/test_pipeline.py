import sys
import os
import unittest
import numpy as np

# Add root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat import registry, register_builtin_plugins
from vol2splat.core.pipeline import Reader, run_pipeline
from vol2splat.core.types import Volume
from vol2splat.config import Config

class DummyReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        # Create a simple 10x10x10 volume
        data = np.zeros((10, 10, 10), dtype=np.float32)
        return Volume(data=data, spacing=(1.0, 1.0, 1.0))

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Register builtins and test components
        register_builtin_plugins()
        try:
            registry.register_reader("dummy_test", DummyReader)
        except:
            pass # Already registered

    def test_basic_flow(self):
        config_dict = {
            "io": {"reader": "dummy_test", "path": "fake.vti"},
            "preprocess": [{"name": "normalize", "method": "minmax"}],
            "sampling": {"name": "uniform", "n_points": 50},
            "export": {"writer": "ply", "path": "test_output.ply"}
        }
        
        cfg = Config.from_dict(config_dict)
        pc = run_pipeline("fake.vti", "test_output.ply", cfg)
        
        self.assertEqual(pc.xyz.shape[0], 50)
        self.assertTrue(os.path.exists("test_output.ply"))
        
        # Cleanup
        if os.path.exists("test_output.ply"):
            os.remove("test_output.ply")

if __name__ == "__main__":
    unittest.main()
