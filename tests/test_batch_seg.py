import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol2splat.batch import assign_cases_to_batches, prepare_case_config_data
from vol2splat.batch_seg import discover_segment_inputs, run_batch_seg


class TestBatchSeg(unittest.TestCase):
    def test_discover_segment_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = os.path.join(tmpdir, "raw")
            organ_dir = os.path.join(raw_root, "s0001", "after_seg_ct")
            os.makedirs(organ_dir, exist_ok=True)
            open(os.path.join(organ_dir, "aorta_density.nii.gz"), "wb").close()
            open(os.path.join(organ_dir, "spinal_cord_density.nii.gz"), "wb").close()

            organs = discover_segment_inputs(
                raw_root,
                reader="nii",
                seg_subdir="after_seg_ct",
                organ_ids="aorta_density",
            )

            self.assertEqual(len(organs), 1)
            self.assertEqual(organs[0]["case_id"], "s0001")
            self.assertEqual(organs[0]["organ_name"], "aorta_density")
            self.assertTrue(organs[0]["input_path"].endswith("aorta_density.nii.gz"))

    def test_prepare_case_config_data_supports_output_name(self):
        organ_case = {
            "case_id": "s0001",
            "output_name": os.path.join("s0001", "aorta_density"),
            "input_path": "/tmp/raw/s0001/after_seg_ct/aorta_density.nii.gz",
            "input_name": "aorta_density.nii.gz",
            "input_stem": "aorta_density",
            "organ_name": "aorta_density",
        }
        assigned = assign_cases_to_batches([organ_case], batch_size=1, shuffle=False)
        base_config = {
            "io": {"reader": "nii", "path": "placeholder.nii.gz"},
            "preprocess": [{"name": "canonicalize", "write_vti": True}],
            "sampling": {"name": "uniform", "n_points": 8},
            "render": {"renderer": "pv_engine", "qc": {"enabled": True}},
            "export": {"writer": "ply", "path": "outputs/placeholder"},
        }

        prepared = prepare_case_config_data(base_config, assigned[0], output_root="/tmp/outputs_seg")

        self.assertTrue(prepared["preprocess"][0]["vti_path"].endswith("s0001/aorta_density/aorta_density_canonical.vti"))
        self.assertTrue(prepared["render"]["path"].endswith("s0001/aorta_density"))
        self.assertTrue(prepared["export"]["path"].endswith("s0001/aorta_density"))

    def test_run_batch_seg_writes_report_with_organ_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = os.path.join(tmpdir, "raw")
            organ_dir = os.path.join(raw_root, "s0001", "after_seg_ct")
            os.makedirs(organ_dir, exist_ok=True)
            open(os.path.join(organ_dir, "aorta_density.nii.gz"), "wb").close()

            config_path = os.path.join(tmpdir, "batch_seg.yaml")
            config_data = {
                "batch": {
                    "size": 10,
                    "shuffle": False,
                    "seed": 7,
                },
                "io": {"reader": "nii", "path": "placeholder.nii.gz"},
                "preprocess": [{"name": "canonicalize", "write_vti": True}],
                "sampling": {"name": "uniform", "n_points": 8},
                "render": {"renderer": "pv_engine", "qc": {"enabled": True}},
                "export": {"writer": "ply", "path": "outputs/placeholder"},
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)

            def fake_run_pipeline_context(_input_path, _output_path, cfg):
                return {
                    "canonical_vti_path": cfg.preprocess[0].params["vti_path"],
                    "written_paths": [os.path.join(cfg.export.path, "points.ply")],
                    "render_result": {
                        "qc": {
                            "failed_tf_names": [],
                            "passed_tf_names": ["TF01"],
                            "report_json": os.path.join(cfg.render.path, "render_qc.json"),
                            "report_md": os.path.join(cfg.render.path, "render_qc.md"),
                        }
                    },
                }

            with patch("vol2splat.batch_seg.run_pipeline_context", side_effect=fake_run_pipeline_context):
                report = run_batch_seg(
                    config_path=config_path,
                    raw_root=raw_root,
                    output_root=os.path.join(tmpdir, "outputs_seg"),
                    seg_subdir="after_seg_ct",
                    case_ids="s0001",
                    organ_ids="aorta_density",
                    shuffle=False,
                )

            self.assertEqual(report["total_cases"], 1)
            self.assertEqual(report["total_organs"], 1)
            self.assertEqual(report["succeeded"], 1)
            self.assertEqual(report["cases"][0]["organ_name"], "aorta_density")
            self.assertTrue(report["cases"][0]["written_paths"][0].endswith("s0001/aorta_density/points.ply"))
            self.assertTrue(os.path.exists(report["report_json"]))

    def test_run_batch_seg_reads_paths_and_filters_from_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, "custom_raw")
            organ_dir = os.path.join(data_root, "s0002", "seg_organs")
            os.makedirs(organ_dir, exist_ok=True)
            open(os.path.join(organ_dir, "aorta_density.nii.gz"), "wb").close()
            open(os.path.join(organ_dir, "spinal_cord_density.nii.gz"), "wb").close()

            config_path = os.path.join(tmpdir, "batch_seg_from_config.yaml")
            config_data = {
                "batch": {
                    "size": 10,
                    "shuffle": False,
                    "seed": 7,
                    "seg": {
                        "raw_root": data_root,
                        "output_root": os.path.join(tmpdir, "custom_outputs"),
                        "case_ids": ["s0002"],
                        "seg_subdir": "seg_organs",
                        "organ_ids": ["spinal_cord_density"],
                    },
                },
                "io": {"reader": "nii", "path": "placeholder.nii.gz"},
                "preprocess": [{"name": "canonicalize", "write_vti": True}],
                "sampling": {"name": "uniform", "n_points": 8},
                "render": {"renderer": "pv_engine", "qc": {"enabled": True}},
                "export": {"writer": "ply", "path": "outputs/placeholder"},
            }
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, sort_keys=False)

            def fake_run_pipeline_context(_input_path, _output_path, cfg):
                return {
                    "canonical_vti_path": cfg.preprocess[0].params["vti_path"],
                    "written_paths": [os.path.join(cfg.export.path, "points.ply")],
                    "render_result": {"qc": {"failed_tf_names": [], "passed_tf_names": ["TF01"]}},
                }

            with patch("vol2splat.batch_seg.run_pipeline_context", side_effect=fake_run_pipeline_context):
                report = run_batch_seg(config_path=config_path)

            self.assertEqual(report["total_cases"], 1)
            self.assertEqual(report["total_organs"], 1)
            self.assertEqual(report["cases"][0]["case_id"], "s0002")
            self.assertEqual(report["cases"][0]["organ_name"], "spinal_cord_density")
            self.assertTrue(report["output_root"].endswith("custom_outputs"))


if __name__ == "__main__":
    unittest.main()
