import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .batch import (
    _default_patterns_for_reader,
    _input_stem,
    _normalize_batch_index,
    _normalize_case_id_list,
    _report_paths,
    assign_cases_to_batches,
    filter_case_infos,
    prepare_case_config_data,
)
from .config import Config, load_config_data
from .core.pipeline import run_pipeline_context


@dataclass(frozen=True)
class SegBatchSettings:
    config_path: str
    raw_root: str
    output_root: str
    reader: str
    case_glob: str
    case_range: str | Sequence[str] | None
    case_start: str | int | None
    case_end: str | int | None
    case_ids: tuple[str, ...]
    exclude_case_ids: tuple[str, ...]
    seg_subdir: str
    organ_glob: str
    organ_ids: tuple[str, ...]
    exclude_organ_ids: tuple[str, ...]
    continue_on_error: bool
    override_paths: bool
    batch_size: int
    shuffle: bool
    seed: int | None
    batch_index: int | None
    batch_prefix: str


def _resolve_option_with_aliases(
    cli_value: Any,
    cfg: Dict[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def _normalize_organ_filters(organ_ids: str | Sequence[str] | None) -> tuple[str, ...]:
    return tuple(_normalize_case_id_list(organ_ids))


def _build_segment_case_info(case_dir: Path, organ_path: Path, seg_subdir: str) -> Dict[str, str]:
    organ_name = _input_stem(str(organ_path))
    case_id = case_dir.name
    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "seg_dir": str((case_dir / seg_subdir).resolve()),
        "input_path": str(organ_path.resolve()),
        "input_name": organ_path.name,
        "input_stem": organ_name,
        "organ_name": organ_name,
        "output_name": os.path.join(case_id, organ_name),
    }


def _filter_organs(
    organ_cases: List[Dict[str, str]],
    organ_ids: str | Sequence[str] | None = None,
    exclude_organ_ids: str | Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    selected_organ_ids = set(_normalize_case_id_list(organ_ids))
    excluded_organ_ids = set(_normalize_case_id_list(exclude_organ_ids))

    filtered: List[Dict[str, str]] = []
    for item in organ_cases:
        organ_name = item["organ_name"]
        if selected_organ_ids and organ_name not in selected_organ_ids:
            continue
        if organ_name in excluded_organ_ids:
            continue
        filtered.append(item)
    return filtered


def discover_segment_inputs(
    raw_root: str,
    reader: str,
    case_glob: str = "s*",
    seg_subdir: str = "after_seg_ct",
    organ_glob: str | None = None,
    case_range: str | Sequence[str] | None = None,
    case_start: str | int | None = None,
    case_end: str | int | None = None,
    case_ids: str | Sequence[str] | None = None,
    exclude_case_ids: str | Sequence[str] | None = None,
    organ_ids: str | Sequence[str] | None = None,
    exclude_organ_ids: str | Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    raw_root_path = Path(raw_root).resolve()
    case_dirs = sorted(path for path in raw_root_path.glob(case_glob) if path.is_dir())
    base_cases = [{"case_id": case_dir.name, "case_dir": str(case_dir)} for case_dir in case_dirs]
    selected_cases = filter_case_infos(
        base_cases,
        case_range=case_range,
        case_start=case_start,
        case_end=case_end,
        case_ids=case_ids,
        exclude_case_ids=exclude_case_ids,
    )

    patterns = [organ_glob] if organ_glob else _default_patterns_for_reader(reader)
    organ_cases: List[Dict[str, str]] = []
    for case in selected_cases:
        case_dir = Path(case["case_dir"])
        seg_dir = case_dir / seg_subdir
        if not seg_dir.is_dir():
            continue

        seen_paths: set[Path] = set()
        for pattern in patterns:
            for organ_path in sorted(seg_dir.glob(pattern)):
                resolved_path = organ_path.resolve()
                if not organ_path.is_file() or resolved_path in seen_paths:
                    continue
                seen_paths.add(resolved_path)
                organ_cases.append(_build_segment_case_info(case_dir, organ_path, seg_subdir))

    return _filter_organs(
        organ_cases,
        organ_ids=organ_ids,
        exclude_organ_ids=exclude_organ_ids,
    )


def _resolve_seg_batch_settings(
    base_config_data: Dict[str, Any],
    config_path: str,
    raw_root: str | None,
    output_root: str | None,
    case_glob: str | None,
    seg_subdir: str | None,
    organ_glob: str | None,
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
    organ_ids: str | Sequence[str] | None,
    exclude_organ_ids: str | Sequence[str] | None,
) -> SegBatchSettings:
    batch_cfg = base_config_data.get("batch") or {}
    seg_batch_cfg = batch_cfg.get("seg") or {}
    reader = base_config_data.get("io", {}).get("reader", "nii")

    resolved_batch_cfg = {**batch_cfg, **seg_batch_cfg}
    resolved_raw_root = _resolve_option_with_aliases(raw_root, resolved_batch_cfg, ("raw_root", "raw-root"), "raw")
    resolved_output_root = _resolve_option_with_aliases(
        output_root,
        resolved_batch_cfg,
        ("output_root", "output-root"),
        "outputs_seg",
    )
    resolved_case_glob = _resolve_option_with_aliases(case_glob, resolved_batch_cfg, ("case_glob", "case-glob"), "s*")

    return SegBatchSettings(
        config_path=os.path.abspath(config_path),
        raw_root=os.path.abspath(resolved_raw_root),
        output_root=os.path.abspath(resolved_output_root),
        reader=reader,
        case_glob=str(resolved_case_glob),
        case_range=_resolve_option_with_aliases(case_range, resolved_batch_cfg, ("case_range", "case-range")),
        case_start=_resolve_option_with_aliases(case_start, resolved_batch_cfg, ("case_start", "case-start")),
        case_end=_resolve_option_with_aliases(case_end, resolved_batch_cfg, ("case_end", "case-end")),
        case_ids=tuple(
            _normalize_case_id_list(_resolve_option_with_aliases(case_ids, resolved_batch_cfg, ("case_ids", "case-ids")))
        ),
        exclude_case_ids=tuple(
            _normalize_case_id_list(
                _resolve_option_with_aliases(exclude_case_ids, resolved_batch_cfg, ("exclude_case_ids", "exclude-case-ids"))
            )
        ),
        seg_subdir=str(_resolve_option_with_aliases(seg_subdir, resolved_batch_cfg, ("seg_subdir", "seg-subdir"), "after_seg_ct")),
        organ_glob=str(_resolve_option_with_aliases(organ_glob, resolved_batch_cfg, ("organ_glob", "organ-glob"), "*.nii.gz")),
        organ_ids=_normalize_organ_filters(
            _resolve_option_with_aliases(organ_ids, resolved_batch_cfg, ("organ_ids", "organ-ids"))
        ),
        exclude_organ_ids=_normalize_organ_filters(
            _resolve_option_with_aliases(exclude_organ_ids, resolved_batch_cfg, ("exclude_organ_ids", "exclude-organ-ids"))
        ),
        continue_on_error=continue_on_error,
        override_paths=override_paths,
        batch_size=int(_resolve_option_with_aliases(batch_size, resolved_batch_cfg, ("size",), 100)),
        shuffle=bool(_resolve_option_with_aliases(shuffle, resolved_batch_cfg, ("shuffle",), True)),
        seed=_resolve_option_with_aliases(seed, resolved_batch_cfg, ("seed",), 0),
        batch_index=_normalize_batch_index(_resolve_option_with_aliases(batch_index, resolved_batch_cfg, ("batch_index", "batch-index"))),
        batch_prefix=str(_resolve_option_with_aliases(None, resolved_batch_cfg, ("prefix",), "batch_")),
    )


def _select_requested_batch(cases: List[Dict[str, str]], batch_index: int | None) -> List[Dict[str, str]]:
    if batch_index is None:
        return cases

    selected_cases = [case for case in cases if case["batch_index"] == batch_index]
    if not selected_cases:
        raise ValueError(f"Requested batch_index={batch_index} but no organs were assigned to that batch")
    return selected_cases


def _initialize_seg_report(
    settings: SegBatchSettings,
    total_batches: int,
    organs: List[Dict[str, str]],
) -> Dict[str, Any]:
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
        "seg_subdir": settings.seg_subdir,
        "organ_glob": settings.organ_glob,
        "organ_ids": list(settings.organ_ids),
        "exclude_organ_ids": list(settings.exclude_organ_ids),
        "batch_size": settings.batch_size,
        "shuffle": settings.shuffle,
        "seed": settings.seed,
        "requested_batch_index": settings.batch_index,
        "total_batches": total_batches,
        "total_cases": len({item["case_id"] for item in organs}),
        "total_organs": len(organs),
        "succeeded": 0,
        "failed": 0,
        "cases": [],
    }


def _write_seg_batch_reports(output_root: str, report: Dict[str, Any]) -> None:
    json_path, md_path = _report_paths(output_root)
    os.makedirs(os.path.abspath(output_root), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Seg Batch Report",
        "",
        f"- total_cases: {report['total_cases']}",
        f"- total_organs: {report['total_organs']}",
        f"- total_batches: {report.get('total_batches', 0)}",
        f"- batch_size: {report.get('batch_size', '-')}",
        f"- shuffle: {report.get('shuffle', '-')}",
        f"- seed: {report.get('seed', '-')}",
        f"- succeeded: {report['succeeded']}",
        f"- failed: {report['failed']}",
        "",
        "| Batch | Case | Organ | Status | Input | QC Failed TF | Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in report["cases"]:
        qc_failed = ",".join(item.get("failed_tf_names", [])) if item.get("failed_tf_names") else "-"
        error = item.get("error", "-")
        lines.append(
            f"| {item.get('batch_id', '-')} | {item['case_id']} | {item.get('organ_name', '-')} | "
            f"{item['status']} | {item['input_name']} | {qc_failed} | {error} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    report["report_json"] = json_path
    report["report_md"] = md_path


def _create_seg_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "batch_id": item.get("batch_id"),
        "batch_index": item.get("batch_index"),
        "batch_case_index": item.get("batch_case_index"),
        "case_id": item["case_id"],
        "organ_name": item["organ_name"],
        "input_name": item["input_name"],
        "input_path": item["input_path"],
        "status": "pending",
    }


def _update_seg_entry_from_success(entry: Dict[str, Any], context: Dict[str, Any]) -> None:
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


def _run_single_segment(
    base_config_data: Dict[str, Any],
    organ_item: Dict[str, Any],
    settings: SegBatchSettings,
) -> tuple[Dict[str, Any], Exception | None]:
    config_data = prepare_case_config_data(
        base_config_data,
        organ_item,
        output_root=settings.output_root,
        override_paths=settings.override_paths,
    )
    cfg = Config.from_dict(config_data)
    entry = _create_seg_entry(organ_item)

    try:
        context = run_pipeline_context(None, None, cfg)
    except Exception as exc:
        entry.update({"status": "failed", "error": str(exc)})
        return entry, exc

    _update_seg_entry_from_success(entry, context)
    return entry, None


def _run_segments(
    base_config_data: Dict[str, Any],
    organs: List[Dict[str, Any]],
    settings: SegBatchSettings,
    report: Dict[str, Any],
) -> None:
    for organ_item in organs:
        entry, error = _run_single_segment(base_config_data, organ_item, settings)
        report["cases"].append(entry)

        if entry["status"] == "ok":
            report["succeeded"] += 1
            continue

        report["failed"] += 1
        if not settings.continue_on_error:
            _write_seg_batch_reports(settings.output_root, report)
            assert error is not None
            raise error


def run_batch_seg(
    config_path: str,
    raw_root: str | None = None,
    output_root: str | None = None,
    case_glob: str | None = None,
    seg_subdir: str | None = None,
    organ_glob: str | None = None,
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
    organ_ids: str | Sequence[str] | None = None,
    exclude_organ_ids: str | Sequence[str] | None = None,
) -> Dict[str, Any]:
    base_config_data = load_config_data(config_path)
    settings = _resolve_seg_batch_settings(
        base_config_data=base_config_data,
        config_path=config_path,
        raw_root=raw_root,
        output_root=output_root,
        case_glob=case_glob,
        seg_subdir=seg_subdir,
        organ_glob=organ_glob,
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
        organ_ids=organ_ids,
        exclude_organ_ids=exclude_organ_ids,
    )

    discovered_organs = discover_segment_inputs(
        settings.raw_root,
        reader=settings.reader,
        case_glob=settings.case_glob,
        seg_subdir=settings.seg_subdir,
        organ_glob=settings.organ_glob,
        case_range=settings.case_range,
        case_start=settings.case_start,
        case_end=settings.case_end,
        case_ids=settings.case_ids,
        exclude_case_ids=settings.exclude_case_ids,
        organ_ids=settings.organ_ids,
        exclude_organ_ids=settings.exclude_organ_ids,
    )
    assigned_organs = assign_cases_to_batches(
        discovered_organs,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        seed=settings.seed,
        batch_prefix=settings.batch_prefix,
    )
    total_batches = max((item["batch_index"] for item in assigned_organs), default=0)
    selected_organs = _select_requested_batch(assigned_organs, settings.batch_index)
    report = _initialize_seg_report(settings, total_batches, selected_organs)

    # Each organ volume is treated as an independent pipeline job.
    _run_segments(base_config_data, selected_organs, settings, report)
    _write_seg_batch_reports(settings.output_root, report)

    json_path, _ = _report_paths(settings.output_root)
    report["report_json"] = json_path
    return report
