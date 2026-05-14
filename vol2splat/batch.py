import copy
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import Config, load_config_data
from .core.pipeline import run_pipeline_context


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


@dataclass(frozen=True)
class BatchSettings:
    # Keep all resolved runtime options together so the execution path can stay simple.
    config_path: str
    raw_root: str
    output_root: str
    reader: str
    case_glob: str
    filename: str | None
    continue_on_error: bool
    override_paths: bool
    batch_size: int
    shuffle: bool
    seed: int | None
    batch_index: int | None
    batch_prefix: str
    case_range: str | Sequence[str] | None
    case_start: str | int | None
    case_end: str | int | None
    case_ids: tuple[str, ...]
    exclude_case_ids: tuple[str, ...]


def _resolve_option_with_aliases(
    cli_value: Any,
    cfg: Dict[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    # Prefer the explicit CLI value, then config aliases, then the built-in default.
    if cli_value is not None:
        return cli_value
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


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


def _normalize_case_id_list(case_ids: str | Sequence[str] | None) -> List[str]:
    if case_ids is None:
        return []
    if isinstance(case_ids, str):
        parts = [item.strip() for item in case_ids.split(",")]
    else:
        parts = [str(item).strip() for item in case_ids]
    return [item for item in parts if item]


def _parse_case_token(value: str | int | None) -> tuple[str, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return ("", value)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return ("", int(text))

    match = re.fullmatch(r"([^\d]*)(\d+)", text)
    if not match:
        return None
    return (match.group(1), int(match.group(2)))


def _parse_case_range(case_range: str | Sequence[str] | None) -> tuple[str | int | None, str | int | None]:
    if case_range is None:
        return (None, None)
    if isinstance(case_range, (list, tuple)):
        if len(case_range) != 2:
            raise ValueError(f"case_range sequence must contain exactly two values, got {case_range!r}")
        return (case_range[0], case_range[1])

    text = str(case_range).strip()
    if not text:
        return (None, None)
    for separator in ("~", "..", ":"):
        if separator in text:
            start, end = text.split(separator, 1)
            return (start.strip() or None, end.strip() or None)
    raise ValueError(f"case_range must look like 's0000~s0100', got {case_range!r}")


def _case_id_in_range(case_id: str, case_start: str | int | None, case_end: str | int | None) -> bool:
    if case_start is None and case_end is None:
        return True

    case_token = _parse_case_token(case_id)
    start_token = _parse_case_token(case_start)
    end_token = _parse_case_token(case_end)

    if case_token and (start_token or end_token):
        case_prefix, case_number = case_token
        if start_token:
            start_prefix, start_number = start_token
            if start_prefix and start_prefix != case_prefix:
                return False
            if case_number < start_number:
                return False
        if end_token:
            end_prefix, end_number = end_token
            if end_prefix and end_prefix != case_prefix:
                return False
            if case_number > end_number:
                return False
        return True

    if case_start is not None and str(case_id) < str(case_start):
        return False
    if case_end is not None and str(case_id) > str(case_end):
        return False
    return True


def filter_case_infos(
    cases: List[Dict[str, str]],
    case_range: str | Sequence[str] | None = None,
    case_start: str | int | None = None,
    case_end: str | int | None = None,
    case_ids: str | Sequence[str] | None = None,
    exclude_case_ids: str | Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    # Apply range/include/exclude filters in one place so discovery stays straightforward.
    range_start, range_end = _parse_case_range(case_range)
    case_start = case_start if case_start is not None else range_start
    case_end = case_end if case_end is not None else range_end

    selected_case_ids = set(_normalize_case_id_list(case_ids))
    excluded_case_ids = set(_normalize_case_id_list(exclude_case_ids))

    filtered: List[Dict[str, str]] = []
    for case in cases:
        case_id = case["case_id"]
        if selected_case_ids and case_id not in selected_case_ids:
            continue
        if case_id in excluded_case_ids:
            continue
        if not _case_id_in_range(case_id, case_start=case_start, case_end=case_end):
            continue
        filtered.append(case)
    return filtered


def _discover_input_for_case(case_dir: Path, patterns: Iterable[str], filename: str | None = None) -> str | None:
    if filename:
        path = case_dir / filename
        return str(path.resolve()) if path.exists() else None

    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(case_dir.glob(pattern)))
    if not matches:
        return None

    unique_matches = sorted({path.resolve() for path in matches}, key=lambda path: str(path))
    preferred = [path for path in unique_matches if path.name in {"ct.nii.gz", "ct.nii", "volume.vti"}]
    chosen_path = preferred[0] if preferred else unique_matches[0]
    return str(chosen_path)


def _build_case_info(case_dir: Path, input_path: str) -> Dict[str, str]:
    case_id = case_dir.name
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "input_path": input_path,
        "input_name": os.path.basename(input_path),
        "input_stem": _input_stem(input_path),
    }


def discover_case_inputs(
    raw_root: str,
    reader: str,
    case_glob: str = "s*",
    filename: str | None = None,
    case_range: str | Sequence[str] | None = None,
    case_start: str | int | None = None,
    case_end: str | int | None = None,
    case_ids: str | Sequence[str] | None = None,
    exclude_case_ids: str | Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    raw_root_path = Path(raw_root).resolve()
    patterns = _default_patterns_for_reader(reader)
    discovered_cases: List[Dict[str, str]] = []

    for case_dir in sorted(path for path in raw_root_path.glob(case_glob) if path.is_dir()):
        input_path = _discover_input_for_case(case_dir, patterns, filename=filename)
        if input_path is None:
            continue
        discovered_cases.append(_build_case_info(case_dir, input_path))

    return filter_case_infos(
        discovered_cases,
        case_range=case_range,
        case_start=case_start,
        case_end=case_end,
        case_ids=case_ids,
        exclude_case_ids=exclude_case_ids,
    )


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


def _build_case_output_context(case_info: Dict[str, str], output_root: str) -> Dict[str, str]:
    output_name = case_info.get("output_name") or case_info["case_id"]
    # Batch assignment is only for scheduling/reporting; per-case outputs stay flat under output_root.
    case_output_dir = os.path.abspath(os.path.join(output_root, output_name))
    batch_output_dir = os.path.abspath(output_root)

    context = dict(case_info)
    context["output_dir"] = case_output_dir
    context["batch_output_dir"] = batch_output_dir
    context["canonical_vti_path"] = os.path.join(case_output_dir, f"{case_info['input_stem']}_canonical.vti")
    context["tiles_dir"] = os.path.join(case_output_dir, "tiles")
    context["export_path"] = case_output_dir
    return context


def _apply_case_specific_paths(config_data: Dict[str, Any], context: Dict[str, str], override_paths: bool) -> None:
    # Apply output defaults after template rendering so config placeholders still work.
    config_data.setdefault("io", {})
    config_data["io"]["path"] = context["input_path"]

    io_tiling = config_data["io"].get("tiling") or config_data["io"].get("tile")
    if isinstance(io_tiling, dict) and (io_tiling.get("write_tiles") or io_tiling.get("output_dir")):
        io_tiling.setdefault("output_dir", os.path.join(context["output_dir"], "input_tiles"))

    for stage in config_data.get("preprocess", []):
        if stage.get("name") != "canonicalize":
            continue
        if override_paths or not stage.get("vti_path"):
            stage["write_vti"] = True
            stage["vti_path"] = context["canonical_vti_path"]

    render_cfg = config_data.get("render")
    if isinstance(render_cfg, dict) and (override_paths or not render_cfg.get("path")):
        render_cfg["path"] = context["output_dir"]
        qc_cfg = render_cfg.get("qc")
        if isinstance(qc_cfg, dict):
            qc_cfg.setdefault("report_json", os.path.join(context["output_dir"], "render_qc.json"))
            qc_cfg.setdefault("report_md", os.path.join(context["output_dir"], "render_qc.md"))

    export_cfg = config_data.get("export")
    if isinstance(export_cfg, dict) and (override_paths or not export_cfg.get("path")):
        export_cfg["path"] = context["export_path"]


def prepare_case_config_data(
    base_config_data: Dict[str, Any],
    case_info: Dict[str, str],
    output_root: str,
    override_paths: bool = True,
) -> Dict[str, Any]:
    context = _build_case_output_context(case_info, output_root)
    config_data = _render_template_values(copy.deepcopy(base_config_data), context)
    _apply_case_specific_paths(config_data, context, override_paths=override_paths)
    return config_data


def _report_paths(output_root: str) -> tuple[str, str]:
    json_path = os.path.join(output_root, "batch_report.json")
    md_path = os.path.join(output_root, "batch_report.md")
    return (os.path.abspath(json_path), os.path.abspath(md_path))


def _sanitize_case_entry_for_report(entry: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(entry)
    sanitized.pop("batch_id", None)
    sanitized.pop("batch_index", None)
    sanitized.pop("batch_case_index", None)
    return sanitized


def _sanitize_report_for_persistence(report: Dict[str, Any]) -> Dict[str, Any]:
    # Keep the file-based report focused on QC results instead of scheduler metadata.
    return {
        "config_path": report["config_path"],
        "raw_root": report["raw_root"],
        "output_root": report["output_root"],
        "case_range": report.get("case_range"),
        "case_start": report.get("case_start"),
        "case_end": report.get("case_end"),
        "succeeded": report["succeeded"],
        "failed": report["failed"],
        "cases": [_sanitize_case_entry_for_report(item) for item in report["cases"]],
    }


def _build_markdown_report_lines(report: Dict[str, Any]) -> List[str]:
    lines = [
        "# Batch Report",
        "",
        f"- succeeded: {report['succeeded']}",
        f"- failed: {report['failed']}",
        "",
        "| Case | Status | Input | Failed TF | Error |",
        "| --- | --- | --- | --- | --- |",
    ]

    for item in report["cases"]:
        qc_failed = ",".join(item.get("failed_tf_names", [])) if item.get("failed_tf_names") else "-"
        error = item.get("error", "-")
        lines.append(
            f"| {item['case_id']} | {item['status']} | {item['input_name']} | {qc_failed} | {error} |"
        )
    return lines


def _write_batch_reports(output_root: str, report: Dict[str, Any]) -> None:
    json_path, md_path = _report_paths(output_root)
    os.makedirs(os.path.abspath(output_root), exist_ok=True)
    persisted_report = _sanitize_report_for_persistence(report)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(persisted_report, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_build_markdown_report_lines(persisted_report)) + "\n")

    report["report_json"] = json_path
    report["report_md"] = md_path


def _resolve_batch_settings(
    base_config_data: Dict[str, Any],
    config_path: str,
    raw_root: str | None,
    output_root: str | None,
    case_glob: str | None,
    filename: str | None,
    continue_on_error: bool,
    override_paths: bool,
    batch_size: int | None,
    shuffle: bool | None,
    seed: int | None,
    batch_index: int | None,
    case_range: str | Sequence[str] | None,
    case_start: str | int | None,
    case_end: str | int | None,
    case_ids: str | Sequence[str] | None,
    exclude_case_ids: str | Sequence[str] | None,
) -> BatchSettings:
    # Resolve config-driven defaults once so the main pipeline only deals with one settings object.
    batch_cfg = base_config_data.get("batch") or {}
    reader = base_config_data.get("io", {}).get("reader", "nii")
    resolved_raw_root = _resolve_option_with_aliases(raw_root, batch_cfg, ("raw_root", "raw-root"), "raw")
    resolved_output_root = _resolve_option_with_aliases(output_root, batch_cfg, ("output_root", "output-root"), "outputs")
    resolved_case_glob = _resolve_option_with_aliases(case_glob, batch_cfg, ("case_glob", "case-glob"), "s*")

    return BatchSettings(
        config_path=os.path.abspath(config_path),
        raw_root=os.path.abspath(str(resolved_raw_root)),
        output_root=os.path.abspath(str(resolved_output_root)),
        reader=reader,
        case_glob=str(resolved_case_glob),
        filename=filename,
        continue_on_error=continue_on_error,
        override_paths=override_paths,
        batch_size=int(_resolve_option_with_aliases(batch_size, batch_cfg, ("size",), 100)),
        shuffle=bool(_resolve_option_with_aliases(shuffle, batch_cfg, ("shuffle",), True)),
        seed=_resolve_option_with_aliases(seed, batch_cfg, ("seed",), 0),
        batch_index=_normalize_batch_index(_resolve_option_with_aliases(batch_index, batch_cfg, ("batch_index", "batch-index"))),
        batch_prefix=str(_resolve_option_with_aliases(None, batch_cfg, ("prefix",), "batch_")),
        case_range=_resolve_option_with_aliases(case_range, batch_cfg, ("case_range", "case-range")),
        case_start=_resolve_option_with_aliases(case_start, batch_cfg, ("case_start", "case-start")),
        case_end=_resolve_option_with_aliases(case_end, batch_cfg, ("case_end", "case-end")),
        case_ids=tuple(_normalize_case_id_list(_resolve_option_with_aliases(case_ids, batch_cfg, ("case_ids", "case-ids")))),
        exclude_case_ids=tuple(
            _normalize_case_id_list(
                _resolve_option_with_aliases(exclude_case_ids, batch_cfg, ("exclude_case_ids", "exclude-case-ids"))
            )
        ),
    )


def _normalize_batch_index(batch_index: int | str | None) -> int | None:
    if batch_index is None:
        return None
    return int(batch_index)


def _discover_and_assign_cases(settings: BatchSettings) -> tuple[List[Dict[str, str]], int]:
    # Discovery and batching are kept separate from execution so the main pipeline stays linear.
    discovered_cases = discover_case_inputs(
        settings.raw_root,
        reader=settings.reader,
        case_glob=settings.case_glob,
        filename=settings.filename,
        case_range=settings.case_range,
        case_start=settings.case_start,
        case_end=settings.case_end,
        case_ids=settings.case_ids,
        exclude_case_ids=settings.exclude_case_ids,
    )
    assigned_cases = assign_cases_to_batches(
        discovered_cases,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        seed=settings.seed,
        batch_prefix=settings.batch_prefix,
    )
    total_batches = max((case["batch_index"] for case in assigned_cases), default=0)
    return assigned_cases, total_batches


def _select_requested_batch(cases: List[Dict[str, str]], batch_index: int | None) -> List[Dict[str, str]]:
    if batch_index is None:
        return cases

    selected_cases = [case for case in cases if case["batch_index"] == batch_index]
    if not selected_cases:
        raise ValueError(f"Requested batch_index={batch_index} but no cases were assigned to that batch")
    return selected_cases


def _initialize_batch_report(settings: BatchSettings, total_batches: int, cases: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "config_path": settings.config_path,
        "raw_root": settings.raw_root,
        "output_root": settings.output_root,
        "case_glob": settings.case_glob,
        "case_range": settings.case_range,
        "case_start": settings.case_start,
        "case_end": settings.case_end,
        "case_ids": list(settings.case_ids),
        "exclude_case_ids": list(settings.exclude_case_ids),
        "batch_size": settings.batch_size,
        "shuffle": settings.shuffle,
        "seed": settings.seed,
        "requested_batch_index": settings.batch_index,
        "total_batches": total_batches,
        "total_cases": len(cases),
        "succeeded": 0,
        "failed": 0,
        "cases": [],
    }


def _create_pending_case_entry(case_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "batch_id": case_info.get("batch_id"),
        "batch_index": case_info.get("batch_index"),
        "batch_case_index": case_info.get("batch_case_index"),
        "case_id": case_info["case_id"],
        "input_name": case_info["input_name"],
        "input_path": case_info["input_path"],
        "status": "pending",
    }


def _update_entry_from_success(entry: Dict[str, Any], context: Dict[str, Any]) -> None:
    render_result = context.get("render_result") or {}
    qc_result = render_result.get("qc") or {}
    entry.update(
        {
            "status": "ok",
            "canonical_vti_path": context.get("canonical_vti_path"),
            "written_paths": context.get("written_paths", []),
            "failed_tf_names": qc_result.get("failed_tf_names", []),
            "passed_tf_names": qc_result.get("passed_tf_names", []),
            "qc_report_json": qc_result.get("report_json"),
            "qc_report_md": qc_result.get("report_md"),
        }
    )


def _run_single_case(
    base_config_data: Dict[str, Any],
    case_info: Dict[str, Any],
    settings: BatchSettings,
) -> tuple[Dict[str, Any], Exception | None]:
    # Keep the per-case execution compact: prepare config, run pipeline, then flatten outputs into one entry.
    config_data = prepare_case_config_data(
        base_config_data,
        case_info,
        output_root=settings.output_root,
        override_paths=settings.override_paths,
    )
    cfg = Config.from_dict(config_data)
    entry = _create_pending_case_entry(case_info)

    try:
        context = run_pipeline_context(None, None, cfg)
    except Exception as exc:
        entry.update({"status": "failed", "error": str(exc)})
        return entry, exc

    _update_entry_from_success(entry, context)
    return entry, None


def _run_cases(
    base_config_data: Dict[str, Any],
    cases: List[Dict[str, Any]],
    settings: BatchSettings,
    report: Dict[str, Any],
) -> None:
    # Update the shared report incrementally so partial progress is preserved on fail-fast runs.
    for case_info in cases:
        entry, error = _run_single_case(base_config_data, case_info, settings)
        report["cases"].append(entry)

        if entry["status"] == "ok":
            report["succeeded"] += 1
            continue

        report["failed"] += 1
        if not settings.continue_on_error:
            _write_batch_reports(settings.output_root, report)
            assert error is not None
            raise error


def run_batch(
    config_path: str,
    raw_root: str | None = None,
    output_root: str | None = None,
    case_glob: str | None = None,
    filename: str | None = None,
    continue_on_error: bool = True,
    override_paths: bool = True,
    batch_size: int | None = None,
    shuffle: bool | None = None,
    seed: int | None = None,
    batch_index: int | None = None,
    case_range: str | Sequence[str] | None = None,
    case_start: str | int | None = None,
    case_end: str | int | None = None,
    case_ids: str | Sequence[str] | None = None,
    exclude_case_ids: str | Sequence[str] | None = None,
) -> Dict[str, Any]:
    # Main flow:
    # 1. resolve config-driven defaults once
    # 2. discover and assign cases
    # 3. run each case through the pipeline
    # 4. persist the final report
    base_config_data = load_config_data(config_path)
    settings = _resolve_batch_settings(
        base_config_data=base_config_data,
        config_path=config_path,
        raw_root=raw_root,
        output_root=output_root,
        case_glob=case_glob,
        filename=filename,
        continue_on_error=continue_on_error,
        override_paths=override_paths,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        batch_index=batch_index,
        case_range=case_range,
        case_start=case_start,
        case_end=case_end,
        case_ids=case_ids,
        exclude_case_ids=exclude_case_ids,
    )

    batch_cfg = base_config_data.get("batch") or {}
    if bool(batch_cfg.get("require_case_ids")) and not settings.case_ids:
        raise ValueError(
            "batch.require_case_ids is true but no case_ids were resolved; set batch.case_ids in YAML or pass --case-ids on the CLI."
        )

    assigned_cases, total_batches = _discover_and_assign_cases(settings)
    selected_cases = _select_requested_batch(assigned_cases, settings.batch_index)
    report = _initialize_batch_report(settings, total_batches, selected_cases)

    # core logic
    _run_cases(base_config_data, selected_cases, settings, report)
    _write_batch_reports(settings.output_root, report)
    return report
