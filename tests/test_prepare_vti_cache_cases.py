import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_vti_cache_cases.py"
    spec = importlib.util.spec_from_file_location("prepare_vti_cache_cases", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestPrepareVtiCacheCases(unittest.TestCase):
    def test_build_vti_case_links_creates_one_case_per_vti(self):
        module = _load_script_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "vti_cache"
            (source_root / "foo").mkdir(parents=True, exist_ok=True)
            (source_root / "bar").mkdir(parents=True, exist_ok=True)
            (source_root / "foo" / "foo_part_0000.vti").write_text("dummy", encoding="utf-8")
            (source_root / "foo" / "foo_part_0001.vti").write_text("dummy", encoding="utf-8")
            (source_root / "bar" / "bar.vti").write_text("dummy", encoding="utf-8")

            output_root = root / "vti_cache_cases"
            report = module.build_vti_case_links(
                source_root=source_root,
                output_root=output_root,
                clean_output_root=True,
            )

            self.assertEqual(report["total_cases"], 3)
            self.assertTrue((output_root / "foo_part_0000" / "input.vti").is_symlink())
            self.assertTrue((output_root / "foo_part_0001" / "input.vti").is_symlink())
            self.assertTrue((output_root / "bar" / "input.vti").is_symlink())
            self.assertTrue((output_root / "_vti_case_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
