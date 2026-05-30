import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vol2splat.batch import assign_cases_to_batches, discover_case_inputs, filter_case_infos, prepare_case_config_data, run_batch
from vol2splat.io.tiling import maybe_tile_input_volume
from vol2splat.rendering.qc import run_render_qc, _apply_metric_rescue_rules
from vol2splat.core.types import Volume

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import torch
except Exception:
    torch = None

try:
    from monai.networks.nets import UNet
except Exception:
    UNet = None

try:
    import vtk  # noqa: F401
except Exception:
    vtk = None


class TestBatchQCTiles(unittest.TestCase):
    def test_foggy_rescue_rule_is_conservative(self):
        rescued_reasons, rescue_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.44,
                "masked_fft_high_freq_ratio": 0.012,
                "detail_over_opacity": 0.058,
            },
            ["foggy_low_detail"],
            {
                "rescue_foggy_foreground_alpha_max": 0.5,
                "rescue_foggy_min_masked_fft_high_freq_ratio": 0.011,
                "rescue_foggy_min_detail_over_opacity": 0.055,
            },
        )
        self.assertEqual(rescued_reasons, [])
        self.assertEqual(rescue_codes, ["rescued_foggy_good_structure"])

        kept_reasons, kept_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.82,
                "masked_fft_high_freq_ratio": 0.028,
                "detail_over_opacity": 0.073,
            },
            ["foggy_low_detail"],
            {
                "rescue_foggy_foreground_alpha_max": 0.5,
                "rescue_foggy_min_masked_fft_high_freq_ratio": 0.011,
                "rescue_foggy_min_detail_over_opacity": 0.055,
            },
        )
        self.assertEqual(kept_reasons, ["foggy_low_detail"])
        self.assertEqual(kept_codes, [])

    def test_peak_structure_rescue_rule_is_selective(self):
        rescued_reasons, rescue_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.54,
                "nonzero_ratio": 0.22,
                "mean_intensity": 0.11,
                "max_masked_fft_high_freq_ratio": 0.0137,
                "max_detail_over_opacity": 0.068,
            },
            ["low_masked_high_frequency", "foggy_low_detail"],
            {
                "rescue_peak_structure_foreground_alpha_max": 0.6,
                "rescue_peak_structure_min_nonzero_ratio": 0.01,
                "rescue_peak_structure_min_mean_intensity": 0.008,
                "rescue_peak_structure_min_max_masked_fft_high_freq_ratio": 0.012,
                "rescue_peak_structure_min_max_detail_over_opacity": 0.06,
            },
        )
        self.assertEqual(rescued_reasons, [])
        self.assertEqual(rescue_codes, ["rescued_peak_structure_soft_foreground"])

        kept_reasons, kept_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.95,
                "nonzero_ratio": 0.22,
                "mean_intensity": 0.11,
                "max_masked_fft_high_freq_ratio": 0.0137,
                "max_detail_over_opacity": 0.068,
            },
            ["low_masked_high_frequency", "foggy_low_detail"],
            {
                "rescue_peak_structure_foreground_alpha_max": 0.6,
                "rescue_peak_structure_min_nonzero_ratio": 0.01,
                "rescue_peak_structure_min_mean_intensity": 0.008,
                "rescue_peak_structure_min_max_masked_fft_high_freq_ratio": 0.012,
                "rescue_peak_structure_min_max_detail_over_opacity": 0.06,
            },
        )
        self.assertEqual(kept_reasons, ["low_masked_high_frequency", "foggy_low_detail"])
        self.assertEqual(kept_codes, [])

        sparse_reasons, sparse_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.45,
                "nonzero_ratio": 0.007,
                "mean_intensity": 0.004,
                "max_masked_fft_high_freq_ratio": 0.19,
                "max_detail_over_opacity": 0.9,
            },
            ["too_sparse", "too_dark"],
            {
                "rescue_peak_structure_foreground_alpha_max": 0.6,
                "rescue_peak_structure_min_nonzero_ratio": 0.01,
                "rescue_peak_structure_min_mean_intensity": 0.008,
                "rescue_peak_structure_min_max_masked_fft_high_freq_ratio": 0.012,
                "rescue_peak_structure_min_max_detail_over_opacity": 0.06,
            },
        )
        self.assertEqual(sparse_reasons, ["too_sparse", "too_dark"])
        self.assertEqual(sparse_codes, [])

    def test_peak_structure_rescue_allowed_reasons_are_configurable(self):
        kept_reasons, kept_codes = _apply_metric_rescue_rules(
            {
                "foreground_alpha_mean": 0.45,
                "nonzero_ratio": 0.08,
                "mean_intensity": 0.04,
                "max_masked_fft_high_freq_ratio": 0.19,
                "max_detail_over_opacity": 0.9,
            },
            ["too_sparse", "too_dark"],
            {
                "rescue_peak_structure_allowed_reasons": [
                    "foggy_low_detail",
                    "low_masked_high_frequency",
                ],
                "rescue_peak_structure_foreground_alpha_max": 0.6,
                "rescue_peak_structure_min_nonzero_ratio": 0.05,
                "rescue_peak_structure_min_mean_intensity": 0.03,
                "rescue_peak_structure_min_max_masked_fft_high_freq_ratio": 0.012,
                "rescue_peak_structure_min_max_detail_over_opacity": 0.06,
            },
        )
        self.assertEqual(kept_reasons, ["too_sparse", "too_dark"])
        self.assertEqual(kept_codes, [])

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
            assigned = assign_cases_to_batches(cases, batch_size=1, shuffle=False)
            prepared = prepare_case_config_data(base_config, assigned[0], output_root=os.path.join(tmpdir, "outputs"))
            self.assertEqual(prepared["io"]["path"], cases[0]["input_path"])
            self.assertEqual(assigned[0]["batch_id"], "batch_0001")
            self.assertTrue(prepared["preprocess"][0]["vti_path"].endswith("s0001/ct_canonical.vti"))
            self.assertTrue(prepared["io"]["tiling"]["output_dir"].endswith("s0001/input_tiles"))
            self.assertTrue(prepared["render"]["path"].endswith("s0001"))
            self.assertTrue(prepared["export"]["path"].endswith("s0001"))

    def test_assign_cases_to_batches_shuffles_repeatably(self):
        cases = [{"case_id": f"s{i:04d}", "input_name": "ct.nii.gz", "input_path": f"/tmp/s{i:04d}/ct.nii.gz", "input_stem": "ct"} for i in range(205)]
        assigned_1 = assign_cases_to_batches(cases, batch_size=100, shuffle=True, seed=7)
        assigned_2 = assign_cases_to_batches(cases, batch_size=100, shuffle=True, seed=7)
        self.assertEqual([case["case_id"] for case in assigned_1], [case["case_id"] for case in assigned_2])
        self.assertEqual(assigned_1[0]["batch_id"], "batch_0001")
        self.assertEqual(assigned_1[99]["batch_id"], "batch_0001")
        self.assertEqual(assigned_1[100]["batch_id"], "batch_0002")
        self.assertEqual(assigned_1[-1]["batch_id"], "batch_0003")

    def test_discover_case_inputs_supports_case_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = os.path.join(tmpdir, "raw")
            for case_id in ("s0001", "s0002", "s0100", "s0101", "x0001"):
                os.makedirs(os.path.join(raw_root, case_id), exist_ok=True)
                open(os.path.join(raw_root, case_id, "ct.nii.gz"), "wb").close()

            cases = discover_case_inputs(raw_root, reader="nii", case_range="s0002~s0100")
            self.assertEqual([case["case_id"] for case in cases], ["s0002", "s0100"])

    def test_filter_case_infos_supports_include_and_exclude_lists(self):
        cases = [
            {"case_id": "s0001", "input_name": "ct.nii.gz", "input_path": "/tmp/s0001/ct.nii.gz", "input_stem": "ct"},
            {"case_id": "s0002", "input_name": "ct.nii.gz", "input_path": "/tmp/s0002/ct.nii.gz", "input_stem": "ct"},
            {"case_id": "s0003", "input_name": "ct.nii.gz", "input_path": "/tmp/s0003/ct.nii.gz", "input_stem": "ct"},
        ]

        filtered = filter_case_infos(
            cases,
            case_ids="s0001,s0002,s9999",
            exclude_case_ids=["s0002"],
        )

        self.assertEqual([case["case_id"] for case in filtered], ["s0001"])

    def test_run_batch_reads_output_root_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = os.path.join(tmpdir, "raw")
            os.makedirs(os.path.join(raw_root, "s0001"), exist_ok=True)
            open(os.path.join(raw_root, "s0001", "ct.nii.gz"), "wb").close()

            config_path = os.path.join(tmpdir, "batch.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch": {
                            "size": 10,
                            "shuffle": False,
                            "seed": 7,
                            "raw_root": raw_root,
                            "output_root": os.path.join(tmpdir, "custom_outputs"),
                        },
                        "io": {"reader": "nii", "path": "placeholder.nii.gz"},
                        "preprocess": [{"name": "canonicalize", "write_vti": True}],
                        "sampling": {"name": "uniform", "n_points": 8},
                        "render": {"renderer": "pv_engine", "qc": {"enabled": True}},
                        "export": {"writer": "ply", "path": "outputs/s0000"},
                    },
                    f,
                )

            def fake_run_pipeline_context(_input_path, _output_path, cfg):
                return {
                    "canonical_vti_path": cfg.preprocess[0].params["vti_path"],
                    "written_paths": [os.path.join(cfg.export.path, "points.ply")],
                    "render_result": {"qc": {"failed_tf_names": [], "passed_tf_names": ["TF01"]}},
                }

            with patch("vol2splat.batch.run_pipeline_context", side_effect=fake_run_pipeline_context):
                report = run_batch(config_path=config_path)

            self.assertTrue(report["output_root"].endswith("custom_outputs"))
            self.assertTrue(report["cases"][0]["written_paths"][0].endswith("custom_outputs/s0001/points.ply"))
            report_json_path = os.path.join(tmpdir, "custom_outputs", "batch_report.json")
            report_md_path = os.path.join(tmpdir, "custom_outputs", "batch_report.md")
            with open(report_json_path, "r", encoding="utf-8") as f:
                persisted_report = json.load(f)
            self.assertNotIn("batch_size", persisted_report)
            self.assertNotIn("total_batches", persisted_report)
            self.assertNotIn("total_cases", persisted_report)
            self.assertNotIn("batch_id", persisted_report["cases"][0])
            with open(report_md_path, "r", encoding="utf-8") as f:
                report_md = f.read()
            self.assertNotIn("batch_size", report_md)
            self.assertNotIn("total_cases", report_md)
            self.assertNotIn("batch_0001", report_md)
            self.assertIn("| Case | Status | Input | Failed TF | Error |", report_md)

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

    @unittest.skipIf(Image is None, "Pillow is required for render QC test")
    def test_render_qc_rejects_foggy_low_frequency_tf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            detail_dir = os.path.join(tmpdir, "TF01", "train")
            foggy_dir = os.path.join(tmpdir, "TF02", "train")
            os.makedirs(detail_dir, exist_ok=True)
            os.makedirs(foggy_dir, exist_ok=True)

            size = 96
            yy, xx = np.mgrid[:size, :size]
            cy = (size - 1) * 0.5
            cx = (size - 1) * 0.5
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask = rr <= size * 0.32

            detail = np.zeros((size, size, 4), dtype=np.uint8)
            checker = (((xx // 4) + (yy // 4)) % 2).astype(np.uint8)
            detail_rgb = 80 + checker * 140
            detail[mask, 0] = detail_rgb[mask]
            detail[mask, 1] = (detail_rgb[mask] * 0.9).astype(np.uint8)
            detail[mask, 2] = (detail_rgb[mask] * 0.7).astype(np.uint8)
            detail[mask, 3] = 255

            foggy = np.zeros((size, size, 4), dtype=np.uint8)
            smooth = np.clip(1.0 - rr / (size * 0.32), 0.0, 1.0)
            smooth_rgb = (70 + smooth * 150).astype(np.uint8)
            foggy[mask, 0] = smooth_rgb[mask]
            foggy[mask, 1] = smooth_rgb[mask]
            foggy[mask, 2] = smooth_rgb[mask]
            foggy[mask, 3] = 255

            Image.fromarray(detail).save(os.path.join(detail_dir, "r_0000.png"))
            Image.fromarray(foggy).save(os.path.join(foggy_dir, "r_0000.png"))

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
                    "min_nonzero_ratio": 0.05,
                    "min_mean_intensity": 0.02,
                    "min_intensity_std": 0.01,
                    "min_masked_fft_high_freq_ratio": 0.02,
                    "min_detail_over_opacity": 0.03,
                },
            )
            self.assertIn("TF01", report["passed_tf_names"])
            self.assertIn("TF02", report["failed_tf_names"])
            failed_item = next(item for item in report["items"] if item["tf_name"] == "TF02")
            self.assertIn("foggy_low_detail", failed_item["reason_codes"])
            self.assertLess(
                failed_item["metrics"]["detail_over_opacity"],
                next(item for item in report["items"] if item["tf_name"] == "TF01")["metrics"]["detail_over_opacity"],
            )

    @unittest.skipIf(Image is None or torch is None or UNet is None, "Pillow, torch and monai are required for segmentation QC test")
    def test_render_qc_with_monai_segmentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = UNet(
                spatial_dims=2,
                in_channels=4,
                out_channels=2,
                channels=(4, 8),
                strides=(1,),
            ).eval()
            model_path = os.path.join(tmpdir, "dummy_monai_unet.pt")
            torch.save(model.state_dict(), model_path)

            bright_dir = os.path.join(tmpdir, "TF01", "train")
            dark_dir = os.path.join(tmpdir, "TF02", "train")
            os.makedirs(bright_dir, exist_ok=True)
            os.makedirs(dark_dir, exist_ok=True)

            bright = np.zeros((16, 16, 4), dtype=np.uint8)
            bright[:, :8, :3] = 255
            bright[:, :, 3] = 255
            dark = np.zeros((16, 16, 4), dtype=np.uint8)
            dark[:, :, 3] = 255
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
                    "segmentation": {
                        "enabled": True,
                        "backend": "monai_unet",
                        "model_path": model_path,
                        "device": "cpu",
                        "input_size": [16, 16],
                        "in_channels": 4,
                        "out_channels": 2,
                        "channels": [4, 8],
                        "strides": [1],
                        "mask_threshold": 0.5,
                        "min_confidence": 0.5,
                        "min_foreground_ratio": 0.1,
                        "min_detected_frames": 1,
                        "min_detected_ratio": 0.5,
                    },
                },
            )
            self.assertEqual(report["items"][0]["metrics"]["segmentation"]["backend"], "monai_unet")
            self.assertIn("segmentation", report["items"][0]["metrics"])

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
