import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import Config, load_config_data
from .core.pipeline import run_pipeline_context


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _input_stem(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _default_patterns_for_reader(reader: str) -> List[str]:
    reader = str(reader).lower()
    if reader in {"nii", "nii.gz", "nifti"}:
        return ["*.nii.gz", "*.nii"]
    if reader == "vti":
        return ["*.vti"]
    if reader == "raw":
        return ["*.raw"]
    return ["*"]


def _discover_input_for_case(case_dir: Path, patterns: Iterable[str], filename: str | None = None) -> str | None:
    if filename:
        path = case_dir / filename
        return str(path.resolve()) if path.exists() else None
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(case_dir.glob(pattern)))
    if not matches:
        return None
    matches = sorted({path.resolve() for path in matches}, key=lambda p: str(p))
    preferred = [path for path in matches if path.name in {"ct.nii.gz", "ct.nii", "volume.vti"}]
    return str((preferred[0] if preferred else matches[0]))


def discover_case_inputs(raw_root: str, reader: str, case_glob: str = "s*", filename: str | None = None) -> List[Dict[str, str]]:
    raw_root_path = Path(raw_root).resolve()
    patterns = _default_patterns_for_reader(reader)
    cases: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in raw_root_path.glob(case_glob) if path.is_dir()):
        input_path = _discover_input_for_case(case_dir, patterns, filename=filename)
        if not input_path:
            continue
        case_id = case_dir.name
        cases.append({
            "case_id": case_id,
            "case_dir": str(case_dir),
            "input_path": input_path,
            "input_name": os.path.basename(input_path),
            "input_stem": _input_stem(input_path),
        })
    return cases


def assign_cases_to_batches(
    cases: List[Dict[str, str]],
    batch_size: int = 100,
    shuffle: bool = True,
    seed: int | None = 0,
    batch_prefix: str = "batch_",
) -> List[Dict[str, str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    assigned = [dict(case) for case in cases]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(assigned)

    total_batches = (len(assigned) + batch_size - 1) // batch_size if assigned else 0
    width = max(4, len(str(total_batches or 1)))
    for index, case in enumerate(assigned):
        batch_index = (index // batch_size) + 1
        case["batch_index"] = batch_index
        case["batch_id"] = f"{batch_prefix}{batch_index:0{width}d}"
        case["batch_case_index"] = (index % batch_size) + 1
        case["batch_size"] = batch_size
        case["batch_shuffle"] = shuffle
        case["batch_seed"] = seed
    return assigned


def _render_template_values(value: Any, context: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormatDict(context))
    if isinstance(value, list):
        return [_render_template_values(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _render_template_values(item, context) for key, item in value.items()}
    return value


def prepare_case_config_data(base_config_data: Dict[str, Any], case_info: Dict[str, str], output_root: str, override_paths: bool = True) -> Dict[str, Any]:
    batch_id = case_info.get("batch_id")
    if batch_id:
        case_output_dir = os.path.abspath(os.path.join(output_root, batch_id, case_info["case_id"]))
        batch_output_dir = os.path.abspath(os.path.join(output_root, batch_id))
    else:
        case_output_dir = os.path.abspath(os.path.join(output_root, case_info["case_id"]))
        batch_output_dir = os.path.abspath(output_root)
    context = dict(case_info)
    context["output_dir"] = case_output_dir
    context["batch_output_dir"] = batch_output_dir
    context["canonical_vti_path"] = os.path.join(case_output_dir, f"{case_info['input_stem']}_canonical.vti")
    context["tiles_dir"] = os.path.join(case_output_dir, "tiles")
    context["export_path"] = case_output_dir

    config_data = _render_template_values(copy.deepcopy(base_config_data), context)
    config_data.setdefault("io", {})
    config_data["io"]["path"] = case_info["input_path"]
    io_tiling = config_data["io"].get("tiling") or config_data["io"].get("tile")
    if isinstance(io_tiling, dict) and (io_tiling.get("write_tiles") or io_tiling.get("output_dir")):
        io_tiling.setdefault("output_dir", os.path.join(case_output_dir, "input_tiles"))

    for stage in config_data.get("preprocess", []):
        if stage.get("name") != "canonicalize":
            continue
        if override_paths or not stage.get("vti_path"):
            stage["write_vti"] = True
            stage["vti_path"] = context["canonical_vti_path"]

    render_cfg = config_data.get("render")
    if isinstance(render_cfg, dict) and (override_paths or not render_cfg.get("path")):
        render_cfg["path"] = case_output_dir
        qc_cfg = render_cfg.get("qc")
        if isinstance(qc_cfg, dict):
            qc_cfg.setdefault("report_json", os.path.join(case_output_dir, "render_qc.json"))
            qc_cfg.setdefault("report_md", os.path.join(case_output_dir, "render_qc.md"))

    export_cfg = config_data.get("export")
    if isinstance(export_cfg, dict) and (override_paths or not export_cfg.get("path")):
        export_cfg["path"] = context["export_path"]
    return config_data


def _write_batch_reports(output_root: str, report: Dict[str, Any]) -> None:
    json_path = os.path.join(output_root, "batch_report.json")
    md_path = os.path.join(output_root, "batch_report.md")
    os.makedirs(os.path.abspath(output_root), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Batch Report",
        "",
        f"- total_cases: {report['total_cases']}",
        f"- total_batches: {report.get('total_batches', 0)}",
        f"- batch_size: {report.get('batch_size', '-')}",
        f"- shuffle: {report.get('shuffle', '-')}",
        f"- seed: {report.get('seed', '-')}",
        f"- succeeded: {report['succeeded']}",
        f"- failed: {report['failed']}",
        "",
        "| Batch | Case | Status | Input | QC Failed TF | Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in report["cases"]:
        qc_failed = ",".join(item.get("failed_tf_names", [])) if item.get("failed_tf_names") else "-"
        error = item.get("error", "-")
        lines.append(f"| {item.get('batch_id', '-')} | {item['case_id']} | {item['status']} | {item['input_name']} | {qc_failed} | {error} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    report["report_json"] = os.path.abspath(json_path)
    report["report_md"] = os.path.abspath(md_path)


def run_batch(
    config_path: str,
    raw_root: str = "raw",
    output_root: str = "outputs",
    case_glob: str = "s*",
    filename: str | None = None,
    continue_on_error: bool = True,
    override_paths: bool = True,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
    batch_index: int | None = None,
) -> Dict[str, Any]:
    base_config_data = load_config_data(config_path)
    batch_cfg = base_config_data.get("batch") or {}
    reader = base_config_data.get("io", {}).get("reader", "nii")
    cases = discover_case_inputs(raw_root, reader=reader, case_glob=case_glob, filename=filename)
    resolved_batch_size = int(batch_size if batch_size is not None else batch_cfg.get("size", 100))
    resolved_shuffle = bool(shuffle if shuffle is not None else batch_cfg.get("shuffle", True))
    resolved_seed = seed if seed is not None else batch_cfg.get("seed", 0)
    selected_batch_index = batch_index if batch_index is not None else batch_cfg.get("batch_index")
    batch_prefix = str(batch_cfg.get("prefix", "batch_"))
    cases = assign_cases_to_batches(
        cases,
        batch_size=resolved_batch_size,
        shuffle=resolved_shuffle,
        seed=resolved_seed,
        batch_prefix=batch_prefix,
    )
    total_batches = max((case["batch_index"] for case in cases), default=0)
    if selected_batch_index is not None:
        selected_batch_index = int(selected_batch_index)
        cases = [case for case in cases if case["batch_index"] == selected_batch_index]
        if not cases:
            raise ValueError(f"Requested batch_index={selected_batch_index} but no cases were assigned to that batch")

    report: Dict[str, Any] = {
        "config_path": os.path.abspath(config_path),
        "raw_root": os.path.abspath(raw_root),
        "output_root": os.path.abspath(output_root),
        "batch_size": resolved_batch_size,
        "shuffle": resolved_shuffle,
        "seed": resolved_seed,
        "requested_batch_index": selected_batch_index,
        "total_batches": total_batches,
        "total_cases": len(cases),
        "succeeded": 0,
        "failed": 0,
        "cases": [],
    }

    for case_info in cases:
        config_data = prepare_case_config_data(base_config_data, case_info, output_root=output_root, override_paths=override_paths)
        cfg = Config.from_dict(config_data)
        entry = {
            "batch_id": case_info.get("batch_id"),
            "batch_index": case_info.get("batch_index"),
            "batch_case_index": case_info.get("batch_case_index"),
            "case_id": case_info["case_id"],
            "input_name": case_info["input_name"],
            "input_path": case_info["input_path"],
            "status": "pending",
        }
        try:
            context = run_pipeline_context(None, None, cfg)
            render_result = context.get("render_result") or {}
            qc_result = render_result.get("qc") or {}
            entry.update({
                "status": "ok",
                "canonical_vti_path": context.get("canonical_vti_path"),
                "written_paths": context.get("written_paths", []),
                "failed_tf_names": qc_result.get("failed_tf_names", []),
                "passed_tf_names": qc_result.get("passed_tf_names", []),
                "qc_report_json": qc_result.get("report_json"),
                "qc_report_md": qc_result.get("report_md"),
            })
            report["succeeded"] += 1
        except Exception as exc:
            entry.update({
                "status": "failed",
                "error": str(exc),
            })
            report["failed"] += 1
            if not continue_on_error:
                report["cases"].append(entry)
                _write_batch_reports(output_root, report)
                raise
        report["cases"].append(entry)

    _write_batch_reports(output_root, report)
    return report
