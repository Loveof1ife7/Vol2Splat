import os
from typing import Any, Dict, Sequence

import numpy as np

from ..core.types import Metadata, Volume, VolumeTile
from ..export.write_vti import write_volume_to_vti


def _parse_block_shape(value: Any, name: str) -> np.ndarray:
    if isinstance(value, (int, float)):
        out = np.array([int(value), int(value), int(value)], dtype=np.int32)
    else:
        if value is None or len(value) != 3:
            raise ValueError(f"{name} must be scalar or length-3 xyz sequence, got {value}")
        out = np.asarray(value, dtype=np.int32)
    if np.any(out <= 0):
        raise ValueError(f"{name} must be positive, got {out.tolist()}")
    return out


def _axis_starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    last_start = length - tile
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(int(v) for v in starts))


def _extract_tiling_cfg(kwargs: Dict[str, Any]) -> Dict[str, Any] | None:
    tiling_cfg = kwargs.get("tiling") or kwargs.get("tile")
    if isinstance(tiling_cfg, dict):
        return dict(tiling_cfg)

    keys = [
        "tile_enabled",
        "tile_size",
        "tile_stride",
        "tile_pad_to_size",
        "tile_pad_value",
        "tile_output_dir",
        "write_tiles",
        "max_voxels",
        "max_dim",
        "max_shape_xyz",
    ]
    extracted = {}
    for key in keys:
        if key in kwargs:
            extracted[key] = kwargs[key]
    return extracted or None


def _normalize_tiling_cfg(cfg: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not cfg:
        return None
    enabled = cfg.get("enabled", cfg.get("tile_enabled", False))
    tile_size = cfg.get("tile_size")
    if not enabled and tile_size is None:
        return None
    return {
        "enabled": bool(enabled or tile_size is not None),
        "tile_size_xyz": _parse_block_shape(tile_size, "tile_size"),
        "tile_stride_xyz": _parse_block_shape(cfg.get("tile_stride", tile_size), "tile_stride"),
        "pad_to_size": bool(cfg.get("tile_pad_to_size", False)),
        "pad_value": float(cfg.get("tile_pad_value", 0.0)),
        "write_tiles": bool(cfg.get("write_tiles", False)),
        "output_dir": cfg.get("output_dir") or cfg.get("tile_output_dir"),
        "max_voxels": int(cfg["max_voxels"]) if cfg.get("max_voxels") is not None else None,
        "max_dim": int(cfg["max_dim"]) if cfg.get("max_dim") is not None else None,
        "max_shape_xyz": _parse_block_shape(cfg["max_shape_xyz"], "max_shape_xyz") if cfg.get("max_shape_xyz") is not None else None,
    }


def _should_tile_volume(vol: Volume, cfg: Dict[str, Any]) -> bool:
    z_dim, y_dim, x_dim = vol.data.shape[-3:]
    shape_xyz = np.array([x_dim, y_dim, z_dim], dtype=np.int64)
    tile_size_xyz = np.asarray(cfg["tile_size_xyz"], dtype=np.int64)
    if np.any(shape_xyz > tile_size_xyz):
        if cfg.get("max_voxels") is None and cfg.get("max_dim") is None and cfg.get("max_shape_xyz") is None:
            return True

    max_voxels = cfg.get("max_voxels")
    if max_voxels is not None and int(np.prod(shape_xyz)) > int(max_voxels):
        return True

    max_dim = cfg.get("max_dim")
    if max_dim is not None and int(shape_xyz.max()) > int(max_dim):
        return True

    max_shape_xyz = cfg.get("max_shape_xyz")
    if max_shape_xyz is not None and np.any(shape_xyz > np.asarray(max_shape_xyz, dtype=np.int64)):
        return True

    return False


def split_volume_into_tiles(vol: Volume, cfg: Dict[str, Any]) -> list[VolumeTile]:
    tile_size_xyz = np.asarray(cfg["tile_size_xyz"], dtype=np.int32)
    tile_stride_xyz = np.asarray(cfg["tile_stride_xyz"], dtype=np.int32)
    tile_size_zyx = np.array([tile_size_xyz[2], tile_size_xyz[1], tile_size_xyz[0]], dtype=np.int32)
    tile_stride_zyx = np.array([tile_stride_xyz[2], tile_stride_xyz[1], tile_stride_xyz[0]], dtype=np.int32)
    z_dim, y_dim, x_dim = vol.data.shape[-3:]
    z_starts = _axis_starts(z_dim, int(tile_size_zyx[0]), int(tile_stride_zyx[0]))
    y_starts = _axis_starts(y_dim, int(tile_size_zyx[1]), int(tile_stride_zyx[1]))
    x_starts = _axis_starts(x_dim, int(tile_size_zyx[2]), int(tile_stride_zyx[2]))

    write_tiles = bool(cfg.get("write_tiles", False))
    output_dir = cfg.get("output_dir")
    pad_to_size = bool(cfg.get("pad_to_size", False))
    pad_value = float(cfg.get("pad_value", 0.0))
    if write_tiles and output_dir:
        os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    tiles: list[VolumeTile] = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1 = min(z0 + int(tile_size_zyx[0]), z_dim)
                y1 = min(y0 + int(tile_size_zyx[1]), y_dim)
                x1 = min(x0 + int(tile_size_zyx[2]), x_dim)
                tile_data = np.asarray(vol.data[z0:z1, y0:y1, x0:x1], dtype=vol.data.dtype)
                if pad_to_size and tile_data.shape != tuple(tile_size_zyx.tolist()):
                    pad_width = (
                        (0, int(tile_size_zyx[0]) - tile_data.shape[0]),
                        (0, int(tile_size_zyx[1]) - tile_data.shape[1]),
                        (0, int(tile_size_zyx[2]) - tile_data.shape[2]),
                    )
                    tile_data = np.pad(tile_data, pad_width, mode="constant", constant_values=float(pad_value))

                tile_origin = vol.origin + np.array([x0, y0, z0], dtype=np.float64) * vol.spacing
                tile_metadata = Metadata(
                    source_path=vol.metadata.source_path,
                    original_dtype=vol.metadata.original_dtype,
                    units=vol.metadata.units,
                    extra=dict(vol.metadata.extra),
                )
                tile_metadata.extra["tile_origin_index_xyz"] = [int(x0), int(y0), int(z0)]
                tile_volume = Volume(
                    data=tile_data,
                    spacing=vol.spacing.copy(),
                    origin=tile_origin,
                    direction=vol.direction.copy(),
                    metadata=tile_metadata,
                )
                tile_name = f"tile_z{z0:04d}_y{y0:04d}_x{x0:04d}"
                tile_path = None
                if write_tiles and output_dir:
                    tile_path = os.path.join(output_dir, f"{tile_name}.vti")
                    write_volume_to_vti(tile_volume, tile_path)
                tiles.append(
                    VolumeTile(
                        name=tile_name,
                        volume=tile_volume,
                        start_xyz=(int(x0), int(y0), int(z0)),
                        path=os.path.abspath(tile_path) if tile_path else None,
                    )
                )
    return tiles


def maybe_tile_input_volume(vol: Volume, kwargs: Dict[str, Any]) -> Volume:
    cfg = _normalize_tiling_cfg(_extract_tiling_cfg(kwargs))
    if not cfg or not cfg.get("enabled", False):
        return vol
    if vol.data.ndim != 3:
        return vol
    if not _should_tile_volume(vol, cfg):
        return vol

    tiles = split_volume_into_tiles(vol, cfg)
    vol.cache["io_tiles"] = tiles
    vol.metadata.extra["io_tiles"] = [
        {
            "name": tile.name,
            "start_xyz": list(tile.start_xyz),
            "path": tile.path,
            "shape_zyx": list(tile.volume.data.shape),
        }
        for tile in tiles
    ]
    return vol
