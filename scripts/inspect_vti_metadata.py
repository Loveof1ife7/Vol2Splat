#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


EXPECTED_NAME_RE = re.compile(
    r"^(?P<name>.+)_(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)_(?P<dtype>[A-Za-z0-9]+)$"
)


@dataclass(frozen=True)
class FileMeta:
    dataset: str
    path: str
    whole_extent: tuple[int, int, int, int, int, int]
    piece_extent: tuple[int, int, int, int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]
    scalar_name: str | None
    scalar_type: str | None
    scalar_format: str | None
    range_min: float | None
    range_max: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan VTI metadata for QC: origin, spacing, extents, direction, scalar type."
    )
    parser.add_argument("root", nargs="?", default=".", help="Root directory to scan.")
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print one line per VTI file in addition to dataset summaries.",
    )
    parser.add_argument(
        "--json-out",
        help="Write full structured output to JSON.",
    )
    parser.add_argument(
        "--csv-out",
        help="Write per-file metadata to CSV.",
    )
    return parser.parse_args()


def parse_extent(text: str) -> tuple[int, int, int, int, int, int]:
    values = tuple(int(v) for v in text.split())
    if len(values) != 6:
        raise ValueError(f"invalid extent: {text!r}")
    return values


def extent_dims(extent: tuple[int, int, int, int, int, int]) -> tuple[int, int, int]:
    return (
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    )


def union_extent(
    extents: Iterable[tuple[int, int, int, int, int, int]]
) -> tuple[int, int, int, int, int, int]:
    extents = list(extents)
    return (
        min(ext[0] for ext in extents),
        max(ext[1] for ext in extents),
        min(ext[2] for ext in extents),
        max(ext[3] for ext in extents),
        min(ext[4] for ext in extents),
        max(ext[5] for ext in extents),
    )


def is_zero_origin(origin: tuple[float, float, float]) -> bool:
    return all(abs(v) < 1e-12 for v in origin)


def is_identity_direction(direction: tuple[float, ...]) -> bool:
    return direction == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def read_vti_header(path: Path, dataset: str) -> FileMeta:
    data = path.read_bytes()
    marker = b"<AppendedData"
    idx = data.find(marker)
    if idx != -1:
        data = data[:idx] + b"</VTKFile>"

    root = ET.fromstring(data.decode("utf-8", errors="ignore"))
    image_data = root.find("ImageData")
    if image_data is None:
        raise ValueError("missing ImageData node")

    piece = image_data.find("Piece")
    if piece is None:
        raise ValueError("missing Piece node")

    data_array = piece.find("./PointData/DataArray")
    scalar_name = data_array.attrib.get("Name") if data_array is not None else None
    scalar_type = data_array.attrib.get("type") if data_array is not None else None
    scalar_format = data_array.attrib.get("format") if data_array is not None else None

    def maybe_float(name: str) -> float | None:
        if data_array is None:
            return None
        value = data_array.attrib.get(name)
        return float(value) if value is not None else None

    return FileMeta(
        dataset=dataset,
        path=str(path),
        whole_extent=parse_extent(image_data.attrib["WholeExtent"]),
        piece_extent=parse_extent(piece.attrib["Extent"]),
        origin=tuple(float(v) for v in image_data.attrib.get("Origin", "0 0 0").split()),
        spacing=tuple(float(v) for v in image_data.attrib.get("Spacing", "1 1 1").split()),
        direction=tuple(
            float(v)
            for v in image_data.attrib.get("Direction", "1 0 0 0 1 0 0 0 1").split()
        ),
        scalar_name=scalar_name,
        scalar_type=scalar_type,
        scalar_format=scalar_format,
        range_min=maybe_float("RangeMin"),
        range_max=maybe_float("RangeMax"),
    )


def expected_from_name(dataset: str) -> dict[str, object] | None:
    match = EXPECTED_NAME_RE.match(dataset)
    if not match:
        return None
    return {
        "dims": (
            int(match.group("nx")),
            int(match.group("ny")),
            int(match.group("nz")),
        ),
        "dtype": match.group("dtype"),
    }


def summarize_dataset(dataset: str, files: list[FileMeta]) -> dict[str, object]:
    expected = expected_from_name(dataset)
    piece_union = union_extent(f.piece_extent for f in files)
    whole_union = union_extent(f.whole_extent for f in files)

    unique_origins = sorted(set(files[i].origin for i in range(len(files))))
    unique_spacings = sorted(set(files[i].spacing for i in range(len(files))))
    unique_directions = sorted(set(files[i].direction for i in range(len(files))))
    unique_scalar_types = sorted(set(f.scalar_type for f in files))
    unique_piece_dims = sorted(set(extent_dims(f.piece_extent) for f in files))
    unique_whole_dims = sorted(set(extent_dims(f.whole_extent) for f in files))

    issues: list[str] = []

    if len(unique_origins) > 1:
        issues.append("inconsistent_origin")
    if len(unique_spacings) > 1:
        issues.append("inconsistent_spacing")
    if len(unique_directions) > 1:
        issues.append("inconsistent_direction")
    if len(unique_scalar_types) > 1:
        issues.append("inconsistent_scalar_type")
    if any(not is_zero_origin(origin) for origin in unique_origins):
        issues.append("non_zero_origin")
    if any(not is_identity_direction(direction) for direction in unique_directions):
        issues.append("non_identity_direction")
    if any(any(v <= 0 for v in spacing) for spacing in unique_spacings):
        issues.append("non_positive_spacing")
    if any(spacing != (1.0, 1.0, 1.0) for spacing in unique_spacings):
        issues.append("non_unit_spacing")

    if expected is not None:
        expected_dims = expected["dims"]
        if tuple(expected_dims) != extent_dims(piece_union):
            issues.append("piece_union_dims_mismatch_name")
        if any(f.scalar_type and f.scalar_type.lower() != str(expected["dtype"]).lower() for f in files):
            issues.append("scalar_type_mismatch_name")

        if len(files) > 1 and tuple(expected_dims) != extent_dims(piece_union):
            local_coords = all(
                is_zero_origin(f.origin)
                and f.piece_extent[0] == 0
                and f.piece_extent[2] == 0
                and f.piece_extent[4] == 0
                for f in files
            )
            if local_coords:
                issues.append("likely_local_partition_metadata")

    return {
        "dataset": dataset,
        "file_count": len(files),
        "expected": expected,
        "piece_union_extent": piece_union,
        "piece_union_dims": extent_dims(piece_union),
        "whole_union_extent": whole_union,
        "whole_union_dims": extent_dims(whole_union),
        "unique_origins": unique_origins,
        "unique_spacings": unique_spacings,
        "unique_directions": unique_directions,
        "unique_scalar_types": unique_scalar_types,
        "unique_piece_dims": unique_piece_dims,
        "unique_whole_dims": unique_whole_dims,
        "issues": issues,
    }


def write_csv(path: Path, files: list[FileMeta]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "path",
                "whole_extent",
                "whole_dims",
                "piece_extent",
                "piece_dims",
                "origin",
                "spacing",
                "direction",
                "scalar_name",
                "scalar_type",
                "scalar_format",
                "range_min",
                "range_max",
            ],
        )
        writer.writeheader()
        for meta in files:
            writer.writerow(
                {
                    "dataset": meta.dataset,
                    "path": meta.path,
                    "whole_extent": meta.whole_extent,
                    "whole_dims": extent_dims(meta.whole_extent),
                    "piece_extent": meta.piece_extent,
                    "piece_dims": extent_dims(meta.piece_extent),
                    "origin": meta.origin,
                    "spacing": meta.spacing,
                    "direction": meta.direction,
                    "scalar_name": meta.scalar_name,
                    "scalar_type": meta.scalar_type,
                    "scalar_format": meta.scalar_format,
                    "range_min": meta.range_min,
                    "range_max": meta.range_max,
                }
            )


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    files = sorted(path.resolve() for path in root.glob("*/*.vti"))
    if not files:
        print(f"no VTI files found under {root}", file=sys.stderr)
        return 1

    metas = [read_vti_header(path, path.parent.name) for path in files]

    by_dataset: dict[str, list[FileMeta]] = {}
    for meta in metas:
        by_dataset.setdefault(meta.dataset, []).append(meta)

    summaries = [summarize_dataset(dataset, files) for dataset, files in sorted(by_dataset.items())]

    print(f"Scanned {len(metas)} VTI files in {len(summaries)} datasets under {root}")
    print()
    for summary in summaries:
        issue_text = ", ".join(summary["issues"]) if summary["issues"] else "OK"
        print(
            f"{summary['dataset']}: files={summary['file_count']}, "
            f"piece_union_dims={summary['piece_union_dims']}, "
            f"spacing={summary['unique_spacings']}, "
            f"origin={summary['unique_origins']}, "
            f"scalar_type={summary['unique_scalar_types']} -> {issue_text}"
        )

    if args.per_file:
        print()
        print("Per-file metadata")
        for meta in metas:
            print(
                f"{meta.path}: whole_dims={extent_dims(meta.whole_extent)}, "
                f"piece_dims={extent_dims(meta.piece_extent)}, "
                f"origin={meta.origin}, spacing={meta.spacing}, type={meta.scalar_type}, "
                f"range=({meta.range_min}, {meta.range_max})"
            )

    if args.json_out:
        payload = {
            "root": str(root),
            "datasets": summaries,
            "files": [
                {
                    **asdict(meta),
                    "whole_dims": extent_dims(meta.whole_extent),
                    "piece_dims": extent_dims(meta.piece_extent),
                }
                for meta in metas
            ],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.csv_out:
        write_csv(Path(args.csv_out), metas)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
