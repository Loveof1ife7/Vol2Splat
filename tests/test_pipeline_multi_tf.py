import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vol2splat import registry, register_builtin_plugins
from vol2splat.config import Config
from vol2splat.core.pipeline import Reader, Renderer, Sampler, Stage, Writer, run_pipeline
from vol2splat.core.types import PointCloud, Volume


class DummyReader(Reader):
    def read(self, path: str, **kwargs) -> Volume:
        return Volume(data=np.zeros((8, 8, 8), dtype=np.float32), spacing=(1.0, 1.0, 1.0))


class MarkCanonicalStage(Stage):
    def run(self, vol: Volume, cfg):
        canonical_vti_path = cfg["canonical_vti_path"]
        os.makedirs(os.path.dirname(canonical_vti_path), exist_ok=True)
        with open(canonical_vti_path, "w", encoding="utf-8") as f:
            f.write("<VTKFile></VTKFile>")
        vol.cache["canonical_vti_path"] = canonical_vti_path
        return vol


class DummyRenderer(Renderer):
    def render(self, canonical_vti_path: str, path: str = None, **kwargs):
        out_dir = os.path.abspath(path)
        tf_outputs = []
        for tf_name in ["TF01", "TF02"]:
            tf_dir = os.path.join(out_dir, tf_name)
            os.makedirs(tf_dir, exist_ok=True)
            tf_json = os.path.join(tf_dir, "tf_config.json")
            with open(tf_json, "w", encoding="utf-8") as f:
                json.dump(
                    {"data_range": [0.0, 1.0], "control_points": [[0.0, 0, 0, 0, 0], [1.0, 1, 1, 1, 1]]},
                    f,
                )
            tf_outputs.append({"tf_name": tf_name, "tf_json": tf_json, "tf_dir": tf_dir})
        return {
            "output_dir": out_dir,
            "canonical_vti_path": canonical_vti_path,
            "tf_outputs": tf_outputs,
            "tf_configs": [item["tf_json"] for item in tf_outputs],
            "render_world_transform": {"scale_factor": 1.0, "offset": [0.0, 0.0, 0.0]},
        }


class DummyOpacitySampler(Sampler):
    seen_tf_json = []

    def sample_canonical_vti(self, canonical_vti_path: str, cfg):
        self.__class__.seen_tf_json.append(cfg["tf_json"])
        xyz = np.zeros((4, 3), dtype=np.float32)
        return PointCloud(xyz=xyz, attrs={"density": np.ones((4, 1), dtype=np.float32)})

    def sample(self, vol: Volume, cfg):
        raise NotImplementedError


class DummyWriter(Writer):
    written_paths = []

    def write(self, pc: PointCloud, path: str, **kwargs):
        self.__class__.written_paths.append(os.path.abspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("dummy")


class TestPipelineMultiTF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_builtin_plugins()
        for name, obj, fn in [
            ("dummy_reader_multi_tf", DummyReader, registry.register_reader),
            ("mark_canonical_multi_tf", MarkCanonicalStage, registry.register_stage),
            ("dummy_renderer_multi_tf", DummyRenderer, registry.register_renderer),
            ("dummy_opacity_multi_tf", DummyOpacitySampler, registry.register_sampler),
            ("dummy_writer_multi_tf", DummyWriter, registry.register_writer),
        ]:
            try:
                fn(name, obj)
            except Exception:
                pass

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="vol2splat_multi_tf_")
        DummyOpacitySampler.seen_tf_json = []
        DummyWriter.written_paths = []

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_render_tf_outputs_are_routed_to_sampling_and_export(self):
        canonical_vti_path = os.path.join(self.tmpdir, "canonical", "case.vti")
        render_dir = os.path.join(self.tmpdir, "render")
        export_path = os.path.join(self.tmpdir, "points.ply")

        cfg = Config.from_dict(
            {
                "io": {"reader": "dummy_reader_multi_tf", "path": "fake.nii.gz"},
                "preprocess": [
                    {
                        "name": "mark_canonical_multi_tf",
                        "canonical_vti_path": canonical_vti_path,
                    }
                ],
                "render": {
                    "renderer": "dummy_renderer_multi_tf",
                    "path": render_dir,
                },
                "sampling": {
                    "name": "dummy_opacity_multi_tf",
                    "sample_all_tf": True,
                },
                "export": {
                    "writer": "dummy_writer_multi_tf",
                    "path": export_path,
                },
            }
        )

        run_pipeline(None, None, cfg)

        self.assertEqual(len(DummyOpacitySampler.seen_tf_json), 2)
        self.assertTrue(os.path.exists(os.path.join(render_dir, "TF01", "points.ply")))
        self.assertTrue(os.path.exists(os.path.join(render_dir, "TF02", "points.ply")))
        self.assertIn(os.path.join(render_dir, "TF01", "points.ply"), DummyWriter.written_paths)
        self.assertIn(os.path.join(render_dir, "TF02", "points.ply"), DummyWriter.written_paths)


if __name__ == "__main__":
    unittest.main()
