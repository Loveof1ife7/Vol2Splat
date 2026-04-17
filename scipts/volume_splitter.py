#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAW -> VTI 切割脚本（Vol2Splat 版本）。

目标：
- 从 manifest 或 volumes 目录读取 raw 数据
- 对任一维度超过 max_dim 的体积进行切块
- 输出到 vti_cache/<dataset_name>/*.vti

命名规则：
- 不切块：<dataset_name>.vti
- 切块：  <dataset_name>_part_0000.vti, _part_0001.vti, ...
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vol2splat.core.types import Metadata, Volume
from vol2splat.export.write_vti import write_volume_to_vti


DEFAULT_MAX_DIM = 512
DATASET_RE = re.compile(r"^(?P<stem>.+)_(?P<X>\d+)x(?P<Y>\d+)x(?P<Z>\d+)_(?P<dtype>uint8|uint16|float32|float64)$", re.I)
MANIFEST_DIM_RE = re.compile(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", re.I)
DTYPE_MAP = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "float32": np.float32,
    "float64": np.float64,
}


# 可按需补充：数据集专用 spacing
SPACING_MAP: Dict[str, Tuple[float, float, float]] = {
    "blunt_fin_256x128x64_uint8": (1.0, 0.75, 1.0),
}

# 可按需补充：数据集专用切块策略 (nx, ny, nz)
# 特殊值 (-1, -1, -1) 表示强制不切块
SPLIT_STRATEGY_MAP: Dict[str, Tuple[int, int, int]] = {
    "jicf_q_1408x1080x1100_float32": (2, 2, 2),
    "magnetic_reconnection_512x512x512_float32": (2, 2, 2),
    "tacc_turbulence_256x256x256_float32": (2, 2, 2),
    "hcci_oh_560x560x560_float32": (-1, -1, -1),
}


def parse_dataset_name(dataset_name: str) -> Tuple[Tuple[int, int, int], str]:
    m = DATASET_RE.match(dataset_name)
    if not m:
        raise ValueError(f"Cannot parse dataset name: {dataset_name}")
    x, y, z = int(m.group("X")), int(m.group("Y")), int(m.group("Z"))
    dtype = m.group("dtype").lower()
    return (x, y, z), dtype


def parse_manifest_line(line: str) -> Optional[Tuple[Path, Tuple[int, int, int], str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 5:
        return None
    dim_text = parts[1]
    dtype_text = parts[2].lower()
    raw_path = Path(parts[4]).expanduser().resolve()
    m = MANIFEST_DIM_RE.match(dim_text)
    if not m:
        return None
    if dtype_text not in DTYPE_MAP:
        return None
    x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return raw_path, (x, y, z), dtype_text


def load_manifest(manifest_path: Path) -> List[Tuple[Path, Tuple[int, int, int], str]]:
    entries: List[Tuple[Path, Tuple[int, int, int], str]] = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_manifest_line(line)
            if not parsed:
                continue
            if str(parsed[0]) in seen:
                continue
            seen.add(str(parsed[0]))
            entries.append(parsed)
    return entries


def scan_volumes_dir(volumes_root: Path) -> List[Tuple[Path, Tuple[int, int, int], str]]:
    entries: List[Tuple[Path, Tuple[int, int, int], str]] = []
    for raw_path in sorted(volumes_root.rglob("*.raw")):
        dataset_name = raw_path.stem
        m = DATASET_RE.match(dataset_name)
        if not m:
            continue
        x, y, z = int(m.group("X")), int(m.group("Y")), int(m.group("Z"))
        dtype = m.group("dtype").lower()
        entries.append((raw_path.resolve(), (x, y, z), dtype))
    return entries


def _chunk_count(dim: int, max_dim: int) -> int:
    return (dim + max_dim - 1) // max_dim


def _chunk_slices(length: int, n_chunks: int) -> List[Tuple[int, int]]:
    assert n_chunks > 0
    base = length // n_chunks
    rem = length % n_chunks
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out


def _write_chunk_to_vti(
    chunk_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    out_path: Path,
    source_path: Path,
) -> None:
    vol = Volume(
        data=np.asarray(chunk_zyx),
        spacing=spacing_xyz,
        origin=(0.0, 0.0, 0.0),
        metadata=Metadata(source_path=str(source_path), original_dtype=str(chunk_zyx.dtype)),
    )
    write_volume_to_vti(vol, str(out_path))


def split_raw_to_vti(
    raw_path: Path,
    shape_xyz: Tuple[int, int, int],
    dtype_name: str,
    vti_cache_root: Path,
    max_dim: int,
) -> List[Path]:
    dataset_name = raw_path.stem
    spacing_xyz = SPACING_MAP.get(dataset_name, (1.0, 1.0, 1.0))
    strategy = SPLIT_STRATEGY_MAP.get(dataset_name)

    x, y, z = shape_xyz
    shape_zyx = (z, y, x)
    data = np.fromfile(raw_path, dtype=DTYPE_MAP[dtype_name])
    expected = int(np.prod(shape_zyx))
    if data.size != expected:
        raise ValueError(f"RAW size mismatch for {raw_path}: expected {expected}, got {data.size}")
    vol_zyx = data.reshape(shape_zyx)

    if strategy == (-1, -1, -1):
        nx = ny = nz = 1
    elif strategy is not None:
        nx, ny, nz = strategy
    else:
        nx = _chunk_count(x, max_dim)
        ny = _chunk_count(y, max_dim)
        nz = _chunk_count(z, max_dim)

    out_dir = (vti_cache_root / dataset_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    x_ranges = _chunk_slices(x, nx)
    y_ranges = _chunk_slices(y, ny)
    z_ranges = _chunk_slices(z, nz)

    written: List[Path] = []
    part_idx = 0
    total_parts = nx * ny * nz
    for z0, z1 in z_ranges:
        for y0, y1 in y_ranges:
            for x0, x1 in x_ranges:
                chunk = vol_zyx[z0:z1, y0:y1, x0:x1].copy()
                if total_parts == 1:
                    out_name = f"{dataset_name}.vti"
                else:
                    out_name = f"{dataset_name}_part_{part_idx:04d}.vti"
                out_path = out_dir / out_name
                _write_chunk_to_vti(chunk, spacing_xyz, out_path, raw_path)
                written.append(out_path)
                part_idx += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Split RAW volumes and write VTI cache files")
    ap.add_argument("--manifest", type=str, default=None, help="Optional manifest file path")
    ap.add_argument("--volumes-root", type=str, default="/root/autodl-tmp/projects/data/volumes")
    ap.add_argument("--vti-cache-root", type=str, default="/root/autodl-tmp/projects/data/vti_cache")
    ap.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    ap.add_argument("--datasets", type=str, default=None, help="Optional comma-separated dataset names")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing VTI files")
    ap.add_argument("--spacing-map-json", type=str, default=None, help="JSON map: dataset_name -> [sx,sy,sz]")
    ap.add_argument("--split-strategy-json", type=str, default=None, help="JSON map: dataset_name -> [nx,ny,nz]")
    args = ap.parse_args()

    if args.max_dim <= 0:
        raise ValueError(f"--max-dim must be > 0, got {args.max_dim}")

    if args.spacing_map_json:
        SPACING_MAP.update({k: tuple(v) for k, v in json.loads(args.spacing_map_json).items()})
    if args.split_strategy_json:
        SPLIT_STRATEGY_MAP.update({k: tuple(v) for k, v in json.loads(args.split_strategy_json).items()})

    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    volumes_root = Path(args.volumes_root).resolve()
    vti_cache_root = Path(args.vti_cache_root).resolve()
    selected = {s.strip() for s in args.datasets.split(",") if s.strip()} if args.datasets else None

    if manifest_path and manifest_path.is_file():
        entries = load_manifest(manifest_path)
    else:
        entries = scan_volumes_dir(volumes_root)

    for raw_path, shape_xyz, dtype_name in entries:
        dataset_name = raw_path.stem
        if selected is not None and dataset_name not in selected:
            continue
        out_dir = vti_cache_root / dataset_name
        if out_dir.exists() and not args.overwrite:
            existing_vti = list(out_dir.glob("*.vti"))
            if existing_vti:
                print(f"[skip] existing VTI cache found: {out_dir}")
                continue

        written = split_raw_to_vti(
            raw_path=raw_path,
            shape_xyz=shape_xyz,
            dtype_name=dtype_name,
            vti_cache_root=vti_cache_root,
            max_dim=args.max_dim,
        )
        for p in written:
            print(f"[ok] {dataset_name} -> {p}")


if __name__ == "__main__":
    main()
