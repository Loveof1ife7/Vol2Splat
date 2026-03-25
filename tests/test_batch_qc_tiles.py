import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat.batch import discover_case_inputs, prepare_case_config_data
from vol2splat.io.tiling import maybe_tile_input_volume
from vol2splat.rendering.qc import run_render_qc
from vol2splat.core.types import Volume

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import vtk  # noqa: F401
except Exception:
    vtk = None


class TestBatchQCTiles(unittest.TestCase):
    def test_discover_case_inputs_and_prepare_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = os.path.join(tmpdir, "raw")
            os.makedirs(os.path.join(raw_root, "s0001"), exist_ok=True)
            os.makedirs(os.path.join(raw_root, "s0002"), exist_ok=True)
            open(os.path.join(raw_root, "s0001", "ct.nii.gz"), "wb").close()
            open(os.path.join(raw_root, "s0002", "ct.nii.gz"), "wb").close()

            cases = discover_case_inputs(raw_root, reader="nii")
            self.assertEqual([case["case_id"] for case in cases], ["s0001", "s0002"])

            base_config = {
                "io": {
                    "reader": "nii",
                    "path": "raw/s0000/ct.nii.gz",
                    "tiling": {"enabled": True, "tile_size": 2, "write_tiles": True},
                },
                "preprocess": [{"name": "canonicalize", "write_vti": True}],
                "sampling": {"name": "uniform", "n_points": 8},
                "render": {"renderer": "pv_engine", "qc": {"enabled": True}},
                "export": {"writer": "ply", "path": "outputs/s0000"},
            }
            prepared = prepare_case_config_data(base_config, cases[0], output_root=os.path.join(tmpdir, "outputs"))
            self.assertEqual(prepared["io"]["path"], cases[0]["input_path"])
            self.assertTrue(prepared["preprocess"][0]["vti_path"].endswith("s0001/ct_canonical.vti"))
            self.assertTrue(prepared["io"]["tiling"]["output_dir"].endswith("s0001/input_tiles"))
            self.assertTrue(prepared["render"]["path"].endswith("s0001"))
            self.assertTrue(prepared["export"]["path"].endswith("s0001"))

    @unittest.skipIf(Image is None, "Pillow is required for render QC test")
    def test_render_qc_marks_dark_tf_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bright_dir = os.path.join(tmpdir, "TF01", "train")
            dark_dir = os.path.join(tmpdir, "TF02", "train")
            os.makedirs(bright_dir, exist_ok=True)
            os.makedirs(dark_dir, exist_ok=True)

            bright = np.zeros((16, 16, 3), dtype=np.uint8)
            bright[:, :8] = 255
            dark = np.zeros((16, 16, 3), dtype=np.uint8)
            Image.fromarray(bright).save(os.path.join(bright_dir, "r_0000.png"))
            Image.fromarray(dark).save(os.path.join(dark_dir, "r_0000.png"))

            tf01_json = os.path.join(tmpdir, "TF01", "tf_config.json")
            tf02_json = os.path.join(tmpdir, "TF02", "tf_config.json")
            with open(tf01_json, "w", encoding="utf-8") as f:
                json.dump({}, f)
            with open(tf02_json, "w", encoding="utf-8") as f:
                json.dump({}, f)

            report = run_render_qc(
                {
                    "output_dir": tmpdir,
                    "tf_outputs": [
                        {"tf_name": "TF01", "tf_json": tf01_json, "tf_dir": os.path.dirname(tf01_json)},
                        {"tf_name": "TF02", "tf_json": tf02_json, "tf_dir": os.path.dirname(tf02_json)},
                    ],
                },
                {
                    "enabled": True,
                    "skip_failed_tf": True,
                    "min_mean_intensity": 0.05,
                    "min_nonzero_ratio": 0.05,
                    "min_intensity_std": 0.001,
                },
            )
            self.assertIn("TF01", report["passed_tf_names"])
            self.assertIn("TF02", report["failed_tf_names"])
            self.assertTrue(os.path.exists(report["report_json"]))
            self.assertTrue(os.path.exists(report["report_md"]))

    @unittest.skipIf(vtk is None, "vtk is required for tile export test")
    def test_io_tiling_writes_tiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vol = Volume(np.arange(64, dtype=np.float32).reshape(4, 4, 4), spacing=(1.0, 1.0, 1.0))
            out = maybe_tile_input_volume(
                vol,
                {
                    "tiling": {
                        "enabled": True,
                        "write_tiles": True,
                        "output_dir": os.path.join(tmpdir, "tiles"),
                        "tile_size": 2,
                    }
                },
            )
            tiles = out.cache.get("io_tiles")
            self.assertEqual(len(tiles), 8)
            self.assertTrue(all(tile.path and os.path.exists(tile.path) for tile in tiles))


if __name__ == "__main__":
    unittest.main()
