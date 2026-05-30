import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "select_top_tf_from_render_qc.py"
    spec = importlib.util.spec_from_file_location("select_top_tf_from_render_qc", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSelectTopTfFromRenderQc(unittest.TestCase):
    def test_select_top_tf_backfills_best_failed_item(self):
        module = _load_script_module()

        report = {
            "items": [
                {
                    "tf_name": "TF02",
                    "status": "fail",
                    "reason_codes": ["foggy_low_detail"],
                    "rescue_codes": [],
                    "metrics": {
                        "mean_intensity": 0.1571,
                        "nonzero_ratio": 0.3543,
                        "intensity_std": 0.2148,
                        "masked_fft_high_freq_ratio": 0.0098,
                        "detail_over_opacity": 0.0367,
                        "max_masked_fft_high_freq_ratio": 0.0176,
                        "max_detail_over_opacity": 0.0775,
                        "foreground_alpha_mean": 0.609,
                    },
                },
                {
                    "tf_name": "TF04",
                    "status": "fail",
                    "reason_codes": ["foggy_low_detail"],
                    "rescue_codes": [],
                    "metrics": {
                        "mean_intensity": 0.1946,
                        "nonzero_ratio": 0.3480,
                        "intensity_std": 0.2688,
                        "masked_fft_high_freq_ratio": 0.0102,
                        "detail_over_opacity": 0.0352,
                        "max_masked_fft_high_freq_ratio": 0.0132,
                        "max_detail_over_opacity": 0.0487,
                        "foreground_alpha_mean": 0.8842,
                    },
                },
                {
                    "tf_name": "TF05",
                    "status": "pass",
                    "reason_codes": [],
                    "rescue_codes": [],
                    "metrics": {
                        "detail_over_opacity": 0.6690,
                        "max_detail_over_opacity": 0.8161,
                        "masked_fft_high_freq_ratio": 0.1043,
                        "max_masked_fft_high_freq_ratio": 0.1211,
                        "foreground_alpha_mean": 0.5304,
                        "intensity_std": 0.2256,
                        "mean_intensity": 0.1003,
                        "nonzero_ratio": 0.1674,
                    },
                },
                {
                    "tf_name": "TF06",
                    "status": "pass",
                    "reason_codes": [],
                    "rescue_codes": [],
                    "metrics": {
                        "detail_over_opacity": 1.8477,
                        "max_detail_over_opacity": 2.2985,
                        "masked_fft_high_freq_ratio": 0.2246,
                        "max_masked_fft_high_freq_ratio": 0.2739,
                        "foreground_alpha_mean": 0.3379,
                        "intensity_std": 0.1400,
                        "mean_intensity": 0.0339,
                        "nonzero_ratio": 0.0568,
                    },
                },
            ]
        }

        selected = module.select_top_tf_items(report, top_k=3)
        self.assertEqual([item["tf_name"] for item in selected], ["TF06", "TF05", "TF02"])

    def test_build_top_tf_symlinks_writes_summary_and_links(self):
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            case_dir = root / "s0703"
            for tf_name in ("TF02", "TF04", "TF05", "TF06"):
                (case_dir / tf_name).mkdir(parents=True, exist_ok=True)

            report = {
                "items": [
                    {
                        "tf_name": "TF02",
                        "status": "fail",
                        "reason_codes": ["foggy_low_detail"],
                        "rescue_codes": [],
                        "metrics": {
                            "detail_over_opacity": 0.0367,
                            "max_detail_over_opacity": 0.0775,
                            "masked_fft_high_freq_ratio": 0.0098,
                            "max_masked_fft_high_freq_ratio": 0.0176,
                            "foreground_alpha_mean": 0.609,
                            "intensity_std": 0.2148,
                            "mean_intensity": 0.1571,
                            "nonzero_ratio": 0.3543,
                        },
                    },
                    {
                        "tf_name": "TF04",
                        "status": "fail",
                        "reason_codes": ["foggy_low_detail"],
                        "rescue_codes": [],
                        "metrics": {
                            "detail_over_opacity": 0.0352,
                            "max_detail_over_opacity": 0.0487,
                            "masked_fft_high_freq_ratio": 0.0102,
                            "max_masked_fft_high_freq_ratio": 0.0132,
                            "foreground_alpha_mean": 0.8842,
                            "intensity_std": 0.2688,
                            "mean_intensity": 0.1946,
                            "nonzero_ratio": 0.3480,
                        },
                    },
                    {
                        "tf_name": "TF05",
                        "status": "pass",
                        "reason_codes": [],
                        "rescue_codes": [],
                        "metrics": {
                            "detail_over_opacity": 0.6690,
                            "max_detail_over_opacity": 0.8161,
                            "masked_fft_high_freq_ratio": 0.1043,
                            "max_masked_fft_high_freq_ratio": 0.1211,
                            "foreground_alpha_mean": 0.5304,
                            "intensity_std": 0.2256,
                            "mean_intensity": 0.1003,
                            "nonzero_ratio": 0.1674,
                        },
                    },
                    {
                        "tf_name": "TF06",
                        "status": "pass",
                        "reason_codes": [],
                        "rescue_codes": [],
                        "metrics": {
                            "detail_over_opacity": 1.8477,
                            "max_detail_over_opacity": 2.2985,
                            "masked_fft_high_freq_ratio": 0.2246,
                            "max_masked_fft_high_freq_ratio": 0.2739,
                            "foreground_alpha_mean": 0.3379,
                            "intensity_std": 0.1400,
                            "mean_intensity": 0.0339,
                            "nonzero_ratio": 0.0568,
                        },
                    },
                ]
            }
            (case_dir / "render_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (case_dir / "render_qc.md").write_text("# dummy\n", encoding="utf-8")

            out = module.build_top_tf_symlinks(
                dataset_root=root,
                output_root=root / "top_tf",
                top_k=3,
                clean_output_root=True,
            )

            self.assertEqual(out["num_cases"], 1)
            self.assertEqual(out["cases"][0]["selected_tf_names"], ["TF06", "TF05", "TF02"])
            self.assertTrue((root / "top_tf" / "s0703" / "TF06").is_symlink())
            self.assertTrue((root / "top_tf" / "s0703" / "TF05").is_symlink())
            self.assertTrue((root / "top_tf" / "s0703" / "TF02").is_symlink())
            self.assertFalse((root / "top_tf" / "s0703" / "TF04").exists())

    def test_main_reads_dataset_root_and_top_k_from_config(self):
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "dataset"
            case_dir = root / "s0001"
            for tf_name in ("TF01", "TF02", "TF03", "TF04"):
                (case_dir / tf_name).mkdir(parents=True, exist_ok=True)

            report = {
                "items": [
                    {"tf_name": "TF01", "status": "pass", "reason_codes": [], "rescue_codes": [], "metrics": {"detail_over_opacity": 0.9, "max_detail_over_opacity": 1.0}},
                    {"tf_name": "TF02", "status": "pass", "reason_codes": [], "rescue_codes": [], "metrics": {"detail_over_opacity": 0.8, "max_detail_over_opacity": 0.9}},
                    {"tf_name": "TF03", "status": "pass", "reason_codes": [], "rescue_codes": [], "metrics": {"detail_over_opacity": 0.7, "max_detail_over_opacity": 0.8}},
                    {"tf_name": "TF04", "status": "pass", "reason_codes": [], "rescue_codes": [], "metrics": {"detail_over_opacity": 0.6, "max_detail_over_opacity": 0.7}},
                ]
            }
            (case_dir / "render_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (case_dir / "render_qc.md").write_text("# dummy\n", encoding="utf-8")

            config_path = Path(tmpdir) / "batch.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "batch:",
                        f"  output_root: {root}",
                        "selection:",
                        "  top_tf:",
                        "    top_k: 2",
                        f"    output_root: {Path(tmpdir) / 'top_tf_cfg'}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            import sys
            from unittest.mock import patch

            with patch.object(sys, "argv", ["select_top_tf_from_render_qc.py", "--config", str(config_path)]):
                module.main()

            out_root = Path(tmpdir) / "top_tf_cfg"
            self.assertTrue((out_root / "s0001" / "TF01").is_symlink())
            self.assertTrue((out_root / "s0001" / "TF02").is_symlink())
            self.assertFalse((out_root / "s0001" / "TF03").exists())


if __name__ == "__main__":
    unittest.main()
