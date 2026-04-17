import typer
from typing import Optional
from pathlib import Path
import sys
import json
from glob import glob

from .batch import run_batch
from .core.pipeline import run_pipeline
from .config import load_config, load_config_data
from .registry import register_builtin_plugins, READERS, STAGES, SAMPLERS, WRITERS, RENDERERS, get_reader
from .core.log import setup_logging
from .rendering.segmentation_qc import evaluate_segmentation_image_paths

app = typer.Typer(help="Vol2Splat: Canonical volume to render/sample/export pipeline")

# Register built-in plugins on import or when app starts?
# Better to do it in callback or main to avoid side effects if just importing app.
# But typer doesn't have a global setup easily.
# We'll do it in the commands or a common init function.

def ensure_plugins():
    register_builtin_plugins()

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
        typer.echo(f"Error: {e}", err=True)
        # Raise to show traceback if verbose? 
        # Typer handles exceptions but we want clean exit for users
        sys.exit(1)


@app.command()
def batch(
    config_path: Path = typer.Option(..., "--config", "-c", help="Configuration file path"),
    raw_root: Path = typer.Option(Path("raw"), "--raw-root", help="Root directory containing case folders"),
    output_root: Path = typer.Option(Path("outputs"), "--output-root", help="Root directory for per-case outputs"),
    input_mode: str = typer.Option("case", "--input-mode", help="Input discovery mode: case or dataset"),
    case_glob: str = typer.Option("s*", "--case-glob", help="Glob used to discover case directories"),
    filename: Optional[str] = typer.Option(None, "--filename", help="Optional fixed input filename inside each case directory"),
    data_root: Optional[Path] = typer.Option(None, "--data-root", help="Unified root for dataset mode; maps to {data-root}/datasets_for_volume_3dgs, {data-root}/vti_cache, {data-root}/volumes"),
    dataset_source_root: Optional[Path] = typer.Option(None, "--dataset-source-root", help="Single source root for dataset mode; auto-detects VTI directories or RAW files"),
    datasets: Optional[str] = typer.Option(None, "--datasets", help="Comma-separated dataset names for dataset mode"),
    datasets_root: Path = typer.Option(Path("datasets_for_volume_3dgs"), "--datasets-root", help="Dataset root used when --datasets is omitted"),
    volumes_root: Path = typer.Option(Path("volumes"), "--volumes-root", help="Raw volume root for dataset mode fallback"),
    vti_cache_root: Path = typer.Option(Path("vti_cache"), "--vti-cache-root", help="VTI cache root for dataset mode"),
    max_vti_parts: Optional[int] = typer.Option(None, "--max-vti-parts", help="Limit VTI parts per dataset in dataset mode"),
    dataset_vti_only: bool = typer.Option(False, "--dataset-vti-only/--dataset-allow-raw-fallback", help="In dataset mode, only use VTI cache and skip datasets without VTI"),
    skip_existing: bool = typer.Option(False, "--skip-existing/--no-skip-existing", help="Skip cases whose output is already complete"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast", help="Continue batch processing when a case fails"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Number of cases per random batch"),
    shuffle: Optional[bool] = typer.Option(None, "--shuffle/--no-shuffle", help="Shuffle cases before grouping into batches"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed used when shuffling cases into batches"),
    batch_index: Optional[int] = typer.Option(None, "--batch-index", help="Only process one 1-based batch index"),
):
    """Run the pipeline for all discovered cases under a raw data root."""
    try:
        if data_root is not None:
            base = data_root.resolve()
            if datasets_root == Path("datasets_for_volume_3dgs"):
                datasets_root = base / "datasets_for_volume_3dgs"
            if vti_cache_root == Path("vti_cache"):
                vti_cache_root = base / "vti_cache"
            if volumes_root == Path("volumes"):
                volumes_root = base / "volumes"
        report = run_batch(
            config_path=str(config_path),
            raw_root=str(raw_root),
            output_root=str(output_root),
            input_mode=input_mode,
            case_glob=case_glob,
            filename=filename,
            datasets=datasets,
            datasets_root=str(datasets_root),
            volumes_root=str(volumes_root),
            vti_cache_root=str(vti_cache_root),
            dataset_source_root=str(dataset_source_root) if dataset_source_root else None,
            max_vti_parts=max_vti_parts,
            dataset_vti_only=dataset_vti_only,
            skip_existing=skip_existing,
            continue_on_error=continue_on_error,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            batch_index=batch_index,
        )
        typer.echo(
            f"Batch completed. batches={report['total_batches']} "
            f"selected={report.get('requested_batch_index') or 'all'} "
            f"cases={report['total_cases']} ok={report['succeeded']} skipped={report.get('skipped', 0)} failed={report['failed']}"
        )
        if report.get("report_json"):
            typer.echo(f"Batch report: {report['report_json']}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)

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
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    app()
