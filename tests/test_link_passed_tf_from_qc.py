import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "link_passed_tf_from_qc.py"
    spec = importlib.util.spec_from_file_location("link_passed_tf_from_qc", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestLinkPassedTfFromQc(unittest.TestCase):
    def test_build_passed_tf_symlinks_mirrors_batch_case_tf_structure(self):
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            case_dir = root / "batch_0001" / "s0001"
            tf01 = case_dir / "TF01"
            tf02 = case_dir / "TF02"
            tf03 = case_dir / "TF03"
            tf01.mkdir(parents=True, exist_ok=True)
            tf02.mkdir(parents=True, exist_ok=True)
            tf03.mkdir(parents=True, exist_ok=True)

            summary = {
                "dataset_root": str(root),
                "cases": [
                    {
                        "batch_id": "batch_0001",
                        "case_id": "s0001",
                        "case_dir": str(case_dir),
                        "passed_tf_names": ["TF01", "TF03"],
                        "failed_tf_names": ["TF02"],
                    }
                ],
            }
            summary_json = root / "qc_lowfreq_summary.json"
            summary_json.parent.mkdir(parents=True, exist_ok=True)
            summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            report = module.build_passed_tf_symlinks(
                summary_json=summary_json,
                output_root=root / "qc",
                clean_output_root=True,
            )

            tf01_link = root / "qc" / "batch_0001" / "s0001" / "TF01"
            tf03_link = root / "qc" / "batch_0001" / "s0001" / "TF03"
            tf02_link = root / "qc" / "batch_0001" / "s0001" / "TF02"

            self.assertTrue(tf01_link.is_symlink())
            self.assertTrue(tf03_link.is_symlink())
            self.assertFalse(tf02_link.exists())
            self.assertEqual(tf01_link.resolve(), tf01.resolve())
            self.assertEqual(tf03_link.resolve(), tf03.resolve())
            self.assertEqual(report["num_passed_tf_links"], 2)
            self.assertTrue((root / "qc" / "passed_tf_links_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
