import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat import register_builtin_plugins, registry
from vol2splat.config import Config
from vol2splat.core.pipeline import Reader, run_pipeline
from vol2splat.core.types import Volume


class DummyReaderForExportDir(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        return Volume(np.zeros((6, 6, 6), dtype=np.float32), spacing=(1.0, 1.0, 1.0))


class TestExportDirPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_builtin_plugins()
        try:
            registry.register_reader("dummy_export_dir", DummyReaderForExportDir)
        except Exception:
            pass

    def test_export_path_directory_gets_default_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config.from_dict(
                {
                    "io": {"reader": "dummy_export_dir", "path": "unused"},
                    "sampling": {"name": "uniform", "n_points": 10},
                    "export": {"writer": "ply", "path": os.path.join(tmpdir, "case_out")},
                }
            )
            run_pipeline(None, None, cfg)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "case_out", "points.ply")))


if __name__ == "__main__":
    unittest.main()
