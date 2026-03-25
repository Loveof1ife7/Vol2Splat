import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat import register_builtin_plugins, registry
from vol2splat.config import Config
from vol2splat.core.pipeline import Reader, run_pipeline_context
from vol2splat.core.types import Volume
from vol2splat.io.tiling import maybe_tile_input_volume


class DummyTiledReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        vol = Volume(np.arange(64, dtype=np.float32).reshape(4, 4, 4), spacing=(1.0, 1.0, 1.0))
        return maybe_tile_input_volume(vol, kwargs)


class TestPipelineIOTiling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_builtin_plugins()
        try:
            registry.register_reader("dummy_tiled_reader", DummyTiledReader)
        except Exception:
            pass

    def test_pipeline_processes_each_io_tile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config.from_dict(
                {
                    "io": {
                        "reader": "dummy_tiled_reader",
                        "path": "unused",
                        "tiling": {"enabled": True, "tile_size": 2},
                    },
                    "sampling": {"name": "uniform", "n_points": 4},
                    "export": {"writer": "ply", "path": os.path.join(tmpdir, "pcd")},
                }
            )
            result = run_pipeline_context(None, None, cfg)
            self.assertEqual(len(result["input_tiles"]), 8)
            self.assertEqual(len(result["tile_results"]), 8)
            expected = os.path.join(tmpdir, "pcd", "tile_z0000_y0000_x0000", "points.ply")
            self.assertTrue(os.path.exists(expected))


if __name__ == "__main__":
    unittest.main()
