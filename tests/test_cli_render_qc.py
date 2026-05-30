import json
import os
import sys
import tempfile
import unittest

import numpy as np
from typer.testing import CliRunner

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol2splat.cli import app

try:
    from PIL import Image
except Exception:
    Image = None


class TestCliRenderQc(unittest.TestCase):
    @unittest.skipIf(Image is None, "Pillow is required for render QC CLI test")
    def test_render_qc_command_runs_on_existing_render_dir(self):
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            render_dir = os.path.join(tmpdir, "s0001")
            bright_dir = os.path.join(render_dir, "TF01", "train")
            dark_dir = os.path.join(render_dir, "TF02", "train")
            os.makedirs(bright_dir, exist_ok=True)
            os.makedirs(dark_dir, exist_ok=True)

            bright = np.zeros((16, 16, 4), dtype=np.uint8)
            bright[:, :8, :3] = 255
            bright[:, :, 3] = 255
            dark = np.zeros((16, 16, 4), dtype=np.uint8)

            Image.fromarray(bright).save(os.path.join(bright_dir, "r_0000.png"))
            Image.fromarray(dark).save(os.path.join(dark_dir, "r_0000.png"))

            for tf_name in ("TF01", "TF02"):
                tf_json = os.path.join(render_dir, tf_name, "tf_config.json")
                with open(tf_json, "w", encoding="utf-8") as f:
                    json.dump({}, f)

            config_path = os.path.join(tmpdir, "qc_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "render": {
                            "qc": {
                                "enabled": True,
                                "skip_failed_tf": True,
                                "min_mean_intensity": 0.05,
                                "min_nonzero_ratio": 0.05,
                                "min_intensity_std": 0.001,
                            }
                        }
                    },
                    f,
                )

            report_json = os.path.join(tmpdir, "render_qc_manual.json")
            report_md = os.path.join(tmpdir, "render_qc_manual.md")
            result = runner.invoke(
                app,
                [
                    "render-qc",
                    "--config",
                    config_path,
                    "--render-dir",
                    render_dir,
                    "--report-json",
                    report_json,
                    "--report-md",
                    report_md,
                ],
            )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Render QC completed.", result.output)
            self.assertTrue(os.path.exists(report_json))
            self.assertTrue(os.path.exists(report_md))

            with open(report_json, "r", encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["passed_tf_names"], ["TF01"])
            self.assertEqual(report["failed_tf_names"], ["TF02"])


if __name__ == "__main__":
    unittest.main()
