import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cmap_variant_canonical_vti.py"
    spec = importlib.util.spec_from_file_location("prepare_cmap_variant_canonical_vti", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPrepareCmapVariantCanonicalVti(unittest.TestCase):
    def test_build_cmap_variant_canonical_vti_copies_anchor_and_uses_relative_symlinks(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical_root = root / "canonical"
            by_cmap_root = root / "by_cmap"
            case_id = "miranda_1024x1024x1024_float32_part_0000"

            src_case = canonical_root / case_id
            src_case.mkdir(parents=True, exist_ok=True)
            src_vti = src_case / f"{case_id}_canonical.vti"
            src_vti.write_text("vtk", encoding="utf-8")

            anchor_case = by_cmap_root / "cool_to_warm" / case_id
            variant_case = by_cmap_root / "turbo" / case_id
            (anchor_case / "TF01").mkdir(parents=True, exist_ok=True)
            (anchor_case / "TF02").mkdir(parents=True, exist_ok=True)
            (variant_case / "TF01").mkdir(parents=True, exist_ok=True)

            report = module.build_cmap_variant_canonical_vti(
                canonical_root=canonical_root,
                by_cmap_root=by_cmap_root,
                anchor_dataset="cool_to_warm",
            )

            anchor_case_vti = anchor_case / "canonical.vti"
            anchor_tf_vti = anchor_case / "TF01" / "canonical.vti"
            variant_case_vti = variant_case / "canonical.vti"
            variant_tf_vti = variant_case / "TF01" / "canonical.vti"

            self.assertEqual(report["num_anchor_case_copies"], 1)
            self.assertEqual(report["num_case_root_links"], 1)
            self.assertEqual(report["num_tf_dir_links"], 3)
            self.assertTrue(anchor_case_vti.is_file())
            self.assertFalse(anchor_case_vti.is_symlink())
            self.assertEqual(anchor_case_vti.read_text(encoding="utf-8"), "vtk")

            self.assertTrue(anchor_tf_vti.is_symlink())
            self.assertEqual(os.readlink(anchor_tf_vti), "../canonical.vti")

            self.assertTrue(variant_case_vti.is_symlink())
            self.assertEqual(
                os.readlink(variant_case_vti),
                os.path.relpath(anchor_case_vti, start=variant_case),
            )

            self.assertTrue(variant_tf_vti.is_symlink())
            self.assertEqual(os.readlink(variant_tf_vti), "../canonical.vti")


if __name__ == "__main__":
    unittest.main()
