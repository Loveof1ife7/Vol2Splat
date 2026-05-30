import typer
from typing import Optional
from pathlib import Path
import sys
import json
from glob import glob

from .batch import run_batch
from .batch_seg import run_batch_seg
from .config_stack import (
    generate_config_stack,
)
from .core.pipeline import run_pipeline
from .config import load_config, load_config_data
from .registry import register_builtin_plugins, READERS, STAGES, SAMPLERS, WRITERS, RENDERERS, get_reader
from .core.log import setup_logging
from .rendering.segmentation_qc import evaluate_segmentation_image_paths
from .rendering.qc import run_render_qc

app = typer.Typer(help="Vol2Splat: Canonical volume to render/sample/export pipeline")

def ensure_plugins():
    # Centralize plugin registration so every command sees the same built-in registry.
    register_builtin_plugins()


def _exit_with_error(exc: Exception) -> None:
    typer.echo(f"Error: {exc}", err=True)
    sys.exit(1)


def _echo_batch_report(report: dict, prefix: str = "Batch completed.") -> None:
    typer.echo(
        f"{prefix} batches={report['total_batches']} "
        f"selected={report.get('requested_batch_index') or 'all'} "
        f"cases={report['total_cases']} ok={report['succeeded']} failed={report['failed']}"
    )
    if report.get("report_json"):
        typer.echo(f"Batch report: {report['report_json']}")


def _run_generated_batch_config(config_path: str) -> dict:
    return run_batch(config_path=config_path)


def _echo_generated_stack(summary: dict) -> None:
    # Show the exact config/output mapping before any long batch execution starts.
    typer.echo(
        f"Generated stack configs: count={len(summary['items'])} "
        f"config_dir={summary['output_dir']} dataset_output_root={summary['dataset_output_root']} "
        f"date_tag={summary['date_tag']}"
    )
    for item in summary["items"]:
        case_scope = (
            f"{item['case_start']}~{item['case_end']}"
            if item.get("case_start") is not None and item.get("case_end") is not None
            else "template-case-scope"
        )
        typer.echo(
            f"  [{item['index']:02d}] {case_scope} "
            f"tf_mode={item['tf_mode']} cmap={item['cmap']} "
            f"output_root={item['output_root']} -> {item['config_path']}"
        )


def _run_generated_stack(summary: dict) -> list[dict]:
    reports: list[dict] = []
    for item in summary["items"]:
        typer.echo(f"Running stack item [{item['index']:02d}] with config {item['config_path']}")
        report = _run_generated_batch_config(item["config_path"])
        _echo_batch_report(report, prefix=f"Stack item [{item['index']:02d}] completed.")
        reports.append(report)
    return reports


def _echo_seg_batch_report(report: dict) -> None:
    typer.echo(
        f"Batch seg completed. batches={report['total_batches']} "
        f"selected={report.get('requested_batch_index') or 'all'} "
        f"cases={report['total_cases']} organs={report['total_organs']} "
        f"ok={report['succeeded']} failed={report['failed']}"
    )
    if report.get("report_json"):
        typer.echo(f"Batch seg report: {report['report_json']}")


def _discover_render_tf_outputs(render_dir: Path) -> list[dict]:
    render_dir = render_dir.resolve()
    tf_outputs: list[dict] = []

    direct_tf_json = render_dir / "tf_config.json"
    if direct_tf_json.is_file():
        tf_outputs.append(
            {
                "tf_name": render_dir.name,
                "tf_json": str(direct_tf_json.resolve()),
                "tf_dir": str(render_dir),
            }
        )

    for tf_json in sorted(render_dir.glob("*/tf_config.json")):
        tf_dir = tf_json.parent.resolve()
        tf_outputs.append(
            {
                "tf_name": tf_dir.name,
                "tf_json": str(tf_json.resolve()),
                "tf_dir": str(tf_dir),
            }
        )
    return tf_outputs

@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")):
    """
    Vol2Splat CLI entry point.
    """
    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")
    
    ensure_plugins()

@app.command()
def run(
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Input volume path"),
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    output_path: Optional[Path] = typer.Option(None, "--output", "-o", help="Output point cloud path"),
):
    """Run the processing pipeline."""
    try:
        cfg = load_config(str(config_path))
        
        run_pipeline(
            str(input_path) if input_path else None, 
            str(output_path) if output_path else None, 
            cfg
        )
        typer.echo("Pipeline completed successfully.")
    except Exception as e:
        _exit_with_error(e)


@app.command()
def batch(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    raw_root: Optional[Path] = typer.Option(None, "--raw-root", help="Root directory containing case folders"),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Root directory for per-case outputs"),
    case_glob: Optional[str] = typer.Option(None, "--case-glob", help="Glob used to discover case directories"),
    case_range: Optional[str] = typer.Option(None, "--case-range", help="Case ID range like s0000~s0100"),
    case_start: Optional[str] = typer.Option(None, "--case-start", help="Inclusive case ID lower bound, e.g. s0000"),
    case_end: Optional[str] = typer.Option(None, "--case-end", help="Inclusive case ID upper bound, e.g. s0100"),
    case_ids: Optional[str] = typer.Option(None, "--case-ids", help="Comma-separated case IDs to include, e.g. s0001,s0007"),
    exclude_case_ids: Optional[str] = typer.Option(None, "--exclude-case-ids", help="Comma-separated case IDs to exclude"),
    filename: Optional[str] = typer.Option(None, "--filename", help="Optional fixed input filename inside each case directory"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast", help="Continue batch processing when a case fails"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Number of cases per random batch"),
    shuffle: Optional[bool] = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle cases before grouping into batches"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed used when shuffling cases into batches"),
    batch_index: Optional[int] = typer.Option(None, "--batch-index", help="Only process one 1-based batch index"),
):
    """Run the pipeline for all discovered cases under a raw data root."""
    try:
        report = run_batch(
            config_path=str(config_path),
            raw_root=str(raw_root) if raw_root else None,
            output_root=str(output_root) if output_root else None,
            case_glob=case_glob,
            case_range=case_range,
            case_start=case_start,
            case_end=case_end,
            case_ids=case_ids,
            exclude_case_ids=exclude_case_ids,
            filename=filename,
            continue_on_error=continue_on_error,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            batch_index=batch_index,
        )
        _echo_batch_report(report)
    except Exception as e:
        _exit_with_error(e)


@app.command("batch-seg")
def batch_seg(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    raw_root: Optional[Path] = typer.Option(None, "--raw-root", help="Root directory containing case folders"),
    output_root: Optional[Path] = typer.Option(None, "--output-root", help="Root directory for per-organ outputs"),
    case_glob: Optional[str] = typer.Option(None, "--case-glob", help="Glob used to discover case directories"),
    seg_subdir: Optional[str] = typer.Option(None, "--seg-subdir", help="Subdirectory under each case containing organ volumes"),
    organ_glob: Optional[str] = typer.Option(None, "--organ-glob", help="Glob used to discover organ files inside seg-subdir"),
    case_range: Optional[str] = typer.Option(None, "--case-range", help="Case ID range like s0000~s0100"),
    case_start: Optional[str] = typer.Option(None, "--case-start", help="Inclusive case ID lower bound, e.g. s0000"),
    case_end: Optional[str] = typer.Option(None, "--case-end", help="Inclusive case ID upper bound, e.g. s0100"),
    case_ids: Optional[str] = typer.Option(None, "--case-ids", help="Comma-separated case IDs to include"),
    exclude_case_ids: Optional[str] = typer.Option(None, "--exclude-case-ids", help="Comma-separated case IDs to exclude"),
    organ_ids: Optional[str] = typer.Option(None, "--organ-ids", help="Comma-separated organ names to include"),
    exclude_organ_ids: Optional[str] = typer.Option(None, "--exclude-organ-ids", help="Comma-separated organ names to exclude"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast", help="Continue batch processing when an organ volume fails"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Number of organ volumes per random batch"),
    shuffle: Optional[bool] = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle organ volumes before grouping into batches"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed used when shuffling organ volumes into batches"),
    batch_index: Optional[int] = typer.Option(None, "--batch-index", help="Only process one 1-based batch index"),
):
    """Run the pipeline for all segmented organ volumes under raw/sxxxx/<seg-subdir>."""
    try:
        report = run_batch_seg(
            config_path=str(config_path),
            raw_root=str(raw_root) if raw_root else None,
            output_root=str(output_root) if output_root else None,
            case_glob=case_glob,
            seg_subdir=seg_subdir,
            organ_glob=organ_glob,
            case_range=case_range,
            case_start=case_start,
            case_end=case_end,
            case_ids=case_ids,
            exclude_case_ids=exclude_case_ids,
            organ_ids=organ_ids,
            exclude_organ_ids=exclude_organ_ids,
            continue_on_error=continue_on_error,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            batch_index=batch_index,
        )
        _echo_seg_batch_report(report)
    except Exception as e:
        _exit_with_error(e)

@app.command("make-config-stack")
def make_config_stack(
    stack_path: Path = typer.Option(..., "--stack", "-s", help="Stack plan YAML path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Optional output directory for generated configs"),
    run_now: bool = typer.Option(False, "--run", help="Run each generated config immediately after generation"),
):
    """Generate multiple batch YAMLs from one stack plan and optionally run them sequentially."""
    try:
        summary = generate_config_stack(
            stack_path=str(stack_path),
            output_dir=str(output_dir) if output_dir else None,
        )
        _echo_generated_stack(summary)
        if run_now:
            _run_generated_stack(summary)
    except Exception as e:
        _exit_with_error(e)

@app.command("list")
def list_plugins():
    """List available plugins."""
    typer.echo("Readers:")
    for name in READERS:
        typer.echo(f"  - {name}")
    typer.echo("\nPreprocess Stages:")
    for name in STAGES:
        typer.echo(f"  - {name}")
    typer.echo("\nSamplers:")
    for name in SAMPLERS:
        typer.echo(f"  - {name}")
    typer.echo("\nRenderers:")
    for name in RENDERERS:
        typer.echo(f"  - {name}")
    typer.echo("\nWriters:")
    for name in WRITERS:
        typer.echo(f"  - {name}")

@app.command()
def inspect(
    input_path: Path = typer.Option(..., "--input", "-i", help="Input volume path"),
    reader: str = typer.Option("vti", "--reader", "-r", help="Reader name (default: vti)"),
):
    """Inspect a volume file."""
    try:
        reader_cls = get_reader(reader)
        r = reader_cls()
        typer.echo(f"Reading {input_path} using {reader}...")
        vol = r.read(str(input_path)) 
        
        typer.echo("--- Volume Info ---")
        typer.echo(f"Shape: {vol.shape}")
        typer.echo(f"Spacing: {vol.spacing}")
        typer.echo(f"Origin: {vol.origin}")
        typer.echo(f"Data Type: {vol.data.dtype}")
        typer.echo(f"Value Range: [{vol.data.min()}, {vol.data.max()}]")
        
    except Exception as e:
        _exit_with_error(e)


@app.command("test-seg")
def test_segmentation(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    tf_dir: Optional[Path] = typer.Option(None, "--tf-dir", help="TF directory such as outputs/s0000/TF01"),
    image_glob: Optional[str] = typer.Option(None, "--image-glob", help="Glob for PNG images to test"),
    split: str = typer.Option("train", "--split", help="Split under tf-dir to evaluate"),
    sample_limit: int = typer.Option(12, "--sample-limit", help="Maximum number of images to evaluate"),
    output_json: Optional[Path] = typer.Option(None, "--output-json", help="Optional output JSON path"),
):
    """Run MONAI segmentation QC on rendered PNGs without running the full pipeline."""
    try:
        config_data = load_config_data(str(config_path))
        segmentation_cfg = (((config_data.get("render") or {}).get("qc") or {}).get("segmentation") or {})
        if not segmentation_cfg.get("enabled", False):
            raise ValueError("render.qc.segmentation is missing or disabled in config")

        if image_glob:
            image_paths = sorted(glob(image_glob))
            base_dir = str(tf_dir) if tf_dir else None
        else:
            if tf_dir is None:
                raise ValueError("Either --tf-dir or --image-glob is required")
            image_paths = sorted(glob(str(tf_dir / split / "*.png")))
            base_dir = str(tf_dir)
        if not image_paths:
            raise FileNotFoundError("No PNG images found for segmentation QC test")

        result = evaluate_segmentation_image_paths(image_paths, segmentation_cfg, sample_limit=sample_limit)
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))

        if output_json is None and base_dir:
            output_json = Path(base_dir) / "segmentation_test.json"
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            typer.echo(f"Wrote segmentation test result to {output_json}")
    except Exception as e:
        _exit_with_error(e)


@app.command("render-qc")
def render_qc(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    render_dir: Path = typer.Option(..., "--render-dir", help="Rendered case directory or a single TF directory"),
    report_json: Optional[Path] = typer.Option(None, "--report-json", help="Optional output JSON path"),
    report_md: Optional[Path] = typer.Option(None, "--report-md", help="Optional output Markdown path"),
):
    """Run render QC on an existing rendered directory without rerunning rendering."""
    try:
        config_data = load_config_data(str(config_path))
        qc_cfg = (((config_data.get("render") or {}).get("qc")) or {})
        if not qc_cfg:
            raise ValueError("render.qc is missing in config")
        if not qc_cfg.get("enabled", True):
            raise ValueError("render.qc is disabled in config")

        render_dir = render_dir.resolve()
        if not render_dir.is_dir():
            raise FileNotFoundError(f"Render directory not found: {render_dir}")

        tf_outputs = _discover_render_tf_outputs(render_dir)
        if not tf_outputs:
            raise FileNotFoundError(f"No tf_config.json found under render directory: {render_dir}")

        qc_cfg = dict(qc_cfg)
        if report_json is not None:
            qc_cfg["report_json"] = str(report_json.resolve())
        if report_md is not None:
            qc_cfg["report_md"] = str(report_md.resolve())

        report = run_render_qc(
            {
                "output_dir": str(render_dir),
                "tf_outputs": tf_outputs,
            },
            qc_cfg,
        )
        typer.echo(
            f"Render QC completed. total_tf={report['total_tf']} "
            f"passed={report['passed_tf']} failed={report['failed_tf']}"
        )
        typer.echo(f"QC JSON: {report['report_json']}")
        typer.echo(f"QC Markdown: {report['report_md']}")
        typer.echo(f"Passed TF: {', '.join(report['passed_tf_names']) if report['passed_tf_names'] else '-'}")
        typer.echo(f"Failed TF: {', '.join(report['failed_tf_names']) if report['failed_tf_names'] else '-'}")
    except Exception as e:
        _exit_with_error(e)

if __name__ == "__main__":
    app()
