import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol2splat.cli import _run_generated_batch_config, _run_generated_stack
from vol2splat.config_stack import (
    build_stack_item_config_data,
    generate_config_stack,
    resolve_cmap_name,
    resolve_tf_mode,
)
from vol2splat.rendering.pv_engine import PVEngineRenderer


class TestConfigStack(unittest.TestCase):
    def test_build_stack_item_config_data_updates_batch_and_render(self):
        template = {
            "batch": {"case_start": "s0000", "case_end": "s0000", "case_range": "s0000~s0000"},
            "render": {"tf_mode": "linear_gs", "cmaps": ["Black, Blue and White"]},
        }

        generated = build_stack_item_config_data(
            template,
            {
                "case_start": "s0001",
                "case_end": "s0100",
                "tf_mode": "linear_gs_sum2",
                "cmap": "Viridis",
            },
        )

        self.assertEqual(generated["batch"]["case_start"], "s0001")
        self.assertEqual(generated["batch"]["case_end"], "s0100")
        self.assertEqual(generated["batch"]["case_range"], "s0001~s0100")
        self.assertEqual(generated["render"]["tf_mode"], "linear_gs_sum2")
        self.assertEqual(generated["render"]["cmaps"], ["Viridis (matplotlib)"])

    def test_resolvers_accept_aliases(self):
        self.assertEqual(resolve_tf_mode("double"), "linear_gs_sum2")
        self.assertEqual(resolve_tf_mode("gaussian"), "gaussian")
        self.assertEqual(resolve_cmap_name("Inferno"), "Inferno (matplotlib)")
        self.assertEqual(resolve_cmap_name("Yellow - Grey - Blue"), "Yellow - Gray - Blue")
        self.assertEqual(resolve_cmap_name("Bluw - Green - Orange"), "Blue - Green - Orange")

    def test_pv_engine_serializes_cmap_lists_as_json(self):
        renderer = PVEngineRenderer()
        flags = renderer._kwargs_to_cli_flags({"cmaps": ["Black, Blue and White", "Turbo"], "num": 84})
        self.assertEqual(flags[0], "--cmaps")
        self.assertEqual(flags[1], '["Black, Blue and White", "Turbo"]')
        self.assertEqual(flags[-2:], ["--num", "84"])

    def test_run_generated_batch_config_delegates_to_batch_runner(self):
        fake_report = {"total_batches": 1, "requested_batch_index": None, "total_cases": 3, "succeeded": 3, "failed": 0}
        with patch("vol2splat.cli.run_batch", return_value=fake_report) as mocked_run_batch:
            report = _run_generated_batch_config("/tmp/generated.yaml")

        mocked_run_batch.assert_called_once_with(config_path="/tmp/generated.yaml")
        self.assertEqual(report, fake_report)

    def test_generate_config_stack_writes_multiple_configs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "batch1_tf1.yaml")
            stack_path = os.path.join(tmpdir, "batch1_stack.yaml")
            with open(template_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "batch": {},
                        "render": {},
                        "io": {"reader": "nii", "path": "raw/{case_id}/ct.nii.gz"},
                        "sampling": {"name": "opacity", "n_points": 1000},
                    },
                    f,
                    sort_keys=False,
                )
            with open(stack_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "template": "batch1_tf1.yaml",
                        "output_dir": "generated_stack",
                        "dataset_output_root": "outputs/generated_stack",
                        "items": [
                            {
                                "name": "first_range",
                                "case_start": "s0000",
                                "case_end": "s0025",
                                "tf_mode": "linear_gs",
                                "cmap": "Yellow - Gray - Blue",
                            },
                            {
                                "name": "second_range",
                                "case_start": "s0026",
                                "case_end": "s0050",
                                "tf_mode": "linear_gs_sum2",
                                "cmap": "Inferno",
                            },
                        ],
                    },
                    f,
                    sort_keys=False,
                )

            summary = generate_config_stack(stack_path=stack_path, date_tag="20260419_120000")

            self.assertEqual(len(summary["items"]), 2)
            self.assertTrue(summary["items"][0]["config_path"].endswith("batch1_tf1_stack_01_first_range_20260419_120000.yaml"))
            with open(summary["items"][1]["config_path"], "r", encoding="utf-8") as f:
                generated = yaml.safe_load(f)
            self.assertEqual(generated["batch"]["case_range"], "s0026~s0050")
            self.assertTrue(generated["batch"]["output_root"].endswith("outputs/generated_stack/second_range"))
            self.assertEqual(generated["render"]["tf_mode"], "linear_gs_sum2")
            self.assertEqual(generated["render"]["cmaps"], ["Inferno (matplotlib)"])

    def test_generate_config_stack_expands_cmaps_and_preserves_template_case_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "batch_sim_render_only.yaml")
            stack_path = os.path.join(tmpdir, "batch_sim_render_only_stack.yaml")
            with open(template_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "batch": {
                            "case_ids": [
                                "miranda_1024x1024x1024_float32_part_0000",
                                "miranda_1024x1024x1024_float32_part_0001",
                            ]
                        },
                        "render": {"tf_mode": "linear_gs_sum2", "cmaps": ["Cool to Warm (Extended)"]},
                        "io": {"reader": "vti", "path": "raw_sim/{case_id}/{case_id}_canonical.vti"},
                    },
                    f,
                    sort_keys=False,
                )
            with open(stack_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "template": "batch_sim_render_only.yaml",
                        "output_dir": "generated_stack",
                        "dataset_output_root": "outputs/generated_stack",
                        "items": [
                            {
                                "name": "miranda_render_only",
                                "tf_mode": "linear_gs_sum2",
                                "cmaps": [
                                    "Cool to Warm (Extended)",
                                    "Turbo",
                                ],
                            }
                        ],
                    },
                    f,
                    sort_keys=False,
                )

            summary = generate_config_stack(stack_path=stack_path, date_tag="20260530_120000")

            self.assertEqual(len(summary["items"]), 2)
            self.assertTrue(summary["items"][0]["name"].startswith("miranda_render_only_"))
            with open(summary["items"][0]["config_path"], "r", encoding="utf-8") as f:
                generated_a = yaml.safe_load(f)
            with open(summary["items"][1]["config_path"], "r", encoding="utf-8") as f:
                generated_b = yaml.safe_load(f)
            self.assertEqual(
                generated_a["batch"]["case_ids"],
                [
                    "miranda_1024x1024x1024_float32_part_0000",
                    "miranda_1024x1024x1024_float32_part_0001",
                ],
            )
            self.assertNotIn("case_range", generated_a["batch"])
            self.assertEqual(generated_a["render"]["cmaps"], ["Cool to Warm (Extended)"])
            self.assertEqual(generated_b["render"]["cmaps"], ["Turbo"])

    def test_generate_config_stack_expands_tf_modes_and_render_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = os.path.join(tmpdir, "batch_sim_render_only.yaml")
            stack_path = os.path.join(tmpdir, "batch_sim_render_only_stack.yaml")
            with open(template_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "batch": {"case_ids": ["miranda_0000"]},
                        "render": {
                            "tf_mode": "linear_gs",
                            "cmaps": ["Cool to Warm (Extended)"],
                            "opaque_unit": 2.0,
                            "opacity_scale": 0.2,
                        },
                        "io": {"reader": "vti", "path": "raw_sim/{case_id}/{case_id}_canonical.vti"},
                    },
                    f,
                    sort_keys=False,
                )
            with open(stack_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "template": "batch_sim_render_only.yaml",
                        "output_dir": "generated_stack",
                        "dataset_output_root": "outputs/generated_stack",
                        "items": [
                            {
                                "name": "miranda_render_sweep",
                                "tf_modes": ["linear_gs_sum2", "gaussian"],
                                "cmaps": ["Turbo", "Jet"],
                                "render_sweep": {
                                    "opaque_unit": [3.0, 7.0],
                                    "opacity_scale": [0.1, 0.3],
                                },
                            }
                        ],
                    },
                    f,
                    sort_keys=False,
                )

            summary = generate_config_stack(stack_path=stack_path, date_tag="20260530_130000")

            self.assertEqual(len(summary["items"]), 16)
            names = [item["name"] for item in summary["items"]]
            self.assertTrue(any("gaussian" in name for name in names))
            self.assertTrue(any("opaque_unit_7_0" in name for name in names))
            self.assertTrue(any("opacity_scale_0_3" in name for name in names))
            with open(summary["items"][-1]["config_path"], "r", encoding="utf-8") as f:
                generated = yaml.safe_load(f)
            self.assertIn(generated["render"]["tf_mode"], ["linear_gs_sum2", "gaussian"])
            self.assertIn(generated["render"]["cmaps"], [["Turbo"], ["Jet"]])
            self.assertIn(generated["render"]["opaque_unit"], [3.0, 7.0])
            self.assertIn(generated["render"]["opacity_scale"], [0.1, 0.3])

    def test_run_generated_stack_delegates_each_config_to_batch_runner(self):
        summary = {
            "items": [
                {"index": 1, "config_path": "/tmp/a.yaml"},
                {"index": 2, "config_path": "/tmp/b.yaml"},
            ]
        }
        fake_report = {"total_batches": 1, "requested_batch_index": None, "total_cases": 1, "succeeded": 1, "failed": 0}

        with patch("vol2splat.cli.run_batch", return_value=fake_report) as mocked_run_batch:
            reports = _run_generated_stack(summary)

        self.assertEqual(mocked_run_batch.call_count, 2)
        self.assertEqual(reports, [fake_report, fake_report])


if __name__ == "__main__":
    unittest.main()
