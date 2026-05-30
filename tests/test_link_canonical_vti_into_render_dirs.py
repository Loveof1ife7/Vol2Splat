import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "link_canonical_vti_into_render_dirs.py"
    spec = importlib.util.spec_from_file_location("link_canonical_vti_into_render_dirs", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestLinkCanonicalVtiIntoRenderDirs(unittest.TestCase):
    def test_build_canonical_vti_symlinks_links_case_root_and_tf_dirs(self):
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            canonical_root = root / "canonical"
            render_root = root / "render"

            case_id = "miranda_1024x1024x1024_float32_part_0000"
            src_case_dir = canonical_root / case_id
            src_case_dir.mkdir(parents=True, exist_ok=True)
            src_vti = src_case_dir / f"{case_id}_canonical.vti"
            src_vti.write_text("<VTKFile></VTKFile>", encoding="utf-8")

            render_case_dir = render_root / case_id
            (render_case_dir / "TF01").mkdir(parents=True, exist_ok=True)
            (render_case_dir / "TF02").mkdir(parents=True, exist_ok=True)

            report = module.build_canonical_vti_symlinks(
                canonical_root=canonical_root,
                render_root=render_root,
                scope="both",
            )

            case_link = render_case_dir / "canonical.vti"
            tf01_link = render_case_dir / "TF01" / "canonical.vti"
            tf02_link = render_case_dir / "TF02" / "canonical.vti"

            self.assertEqual(report["num_cases"], 1)
            self.assertEqual(report["num_links"], 3)
            self.assertTrue(case_link.is_symlink())
            self.assertTrue(tf01_link.is_symlink())
            self.assertTrue(tf02_link.is_symlink())
            self.assertEqual(os.path.realpath(case_link), str(src_vti.resolve()))
            self.assertEqual(os.path.realpath(tf01_link), str(src_vti.resolve()))
            self.assertEqual(os.path.realpath(tf02_link), str(src_vti.resolve()))


if __name__ == "__main__":
    unittest.main()
