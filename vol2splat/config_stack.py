from __future__ import annotations

import copy
from datetime import datetime
from itertools import product
from pathlib import Path
import re
from typing import Any, Dict

import yaml

from .config import load_config_data


TF_MODE_PRESETS = {
    "single": "linear_gs",
    "double": "linear_gs_sum2",
    "unit": "unit",
    "linear": "linear",
    "linear_sum2": "linear_sum2",
    "linear_pair": "linear_pair",
    "linear_soft": "linear_soft",
    "linear_soft_sum2": "linear_soft_sum2",
    "linear_vol2splat": "linear_vol2splat",
    "linear_v2splat": "linear_v2splat",
    "linear_vol2splat_sum2": "linear_vol2splat_sum2",
    "linear_v2splat_sum2": "linear_v2splat_sum2",
    "gaussian": "gaussian",
    "gaussian_sum2": "gaussian_sum2",
    "gaussian_pair": "gaussian_pair",
    "linear_gs": "linear_gs",
    "linear_gsdatagen": "linear_gsdatagen",
    "linear_gs_sum2": "linear_gs_sum2",
    "linear_gs_pair": "linear_gs_pair",
}

CMAP_PRESETS = {
    "Viridis": "Viridis (matplotlib)",
    "Turbo": "Turbo",
    "Cool to Warm (Extended)": "Cool to Warm (Extended)",
    "Cool to Warm": "Cool to Warm",
    "Grayscale": "Grayscale",
    "Inferno": "Inferno (matplotlib)",
    "Blue Orange (divergent)": "Blue Orange (divergent)",
    "Blue - Green - Orange": "Blue - Green - Orange",
    "Bluw - Green - Orange": "Blue - Green - Orange",
    "Rainbow Uniform": "Rainbow Uniform",
    "Rainbow Desaturated": "Rainbow Desaturated",
    "Yellow - Gray - Blue": "Yellow - Gray - Blue",
    "Yellow - Grey - Blue": "Yellow - Gray - Blue",
    "Black, Blue and White": "Black, Blue and White",
    "Jet": "Jet",
}


def resolve_tf_mode(value: str) -> str:
    key = str(value).strip()
    if key not in TF_MODE_PRESETS:
        raise ValueError(f"Unsupported tf_mode={value!r}; expected one of {', '.join(TF_MODE_PRESETS)}")
    return TF_MODE_PRESETS[key]


def resolve_cmap_name(value: str) -> str:
    key = str(value).strip()
    if key in CMAP_PRESETS:
        return CMAP_PRESETS[key]
    if key in CMAP_PRESETS.values():
        return key
    raise ValueError(f"Unsupported cmap={value!r}; expected one of {', '.join(CMAP_PRESETS)}")


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return text.strip("_").lower() or "item"


def _clone_value(value: Any) -> Any:
    return copy.deepcopy(value)


def _update_batch_section(batch_cfg: Dict[str, Any], case_start: str, case_end: str) -> None:
    batch_cfg["case_start"] = case_start
    batch_cfg["case_end"] = case_end
    batch_cfg["case_range"] = f"{case_start}~{case_end}"


def _normalize_render_value(key: str, value: Any) -> Any:
    if key == "tf_mode":
        return resolve_tf_mode(str(value))
    if key == "cmaps":
        if isinstance(value, list):
            return [resolve_cmap_name(str(item)) for item in value]
        return [resolve_cmap_name(str(value))]
    if key == "cmap":
        return [resolve_cmap_name(str(value))]
    return _clone_value(value)


def _set_render_param(render_cfg: Dict[str, Any], key: str, value: Any) -> None:
    normalized = _normalize_render_value(key, value)
    if key == "cmap":
        render_cfg["cmaps"] = normalized
        return
    render_cfg[key] = normalized


def _apply_render_overrides(render_cfg: Dict[str, Any], overrides: Dict[str, Any] | None) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        _set_render_param(render_cfg, str(key), value)


def build_stack_item_config_data(template_data: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    config_data = copy.deepcopy(template_data)
    batch_cfg = config_data.setdefault("batch", {})
    render_cfg = config_data.setdefault("render", {})
    case_start = item.get("case_start")
    case_end = item.get("case_end")
    if case_start is not None or case_end is not None:
        if case_start is None or case_end is None:
            raise ValueError("Stack item must provide both case_start and case_end when either is set")
        _update_batch_section(batch_cfg, case_start=str(case_start), case_end=str(case_end))
    _apply_render_overrides(
        render_cfg,
        {
            "tf_mode": item["tf_mode"],
            "cmap": item["cmap"],
        },
    )
    _apply_render_overrides(render_cfg, item.get("render_overrides"))
    return config_data


def _render_sweep_axes_from_item(item: Dict[str, Any]) -> list[tuple[str, list[Any]]]:
    axes: list[tuple[str, list[Any]]] = []
    if item.get("cmaps") is not None:
        cmaps = item.get("cmaps")
        if not isinstance(cmaps, list) or not cmaps:
            raise ValueError(f"Stack item cmaps must be a non-empty list, got {cmaps!r}")
        axes.append(("cmap", list(cmaps)))
    if item.get("tf_modes") is not None:
        tf_modes = item.get("tf_modes")
        if not isinstance(tf_modes, list) or not tf_modes:
            raise ValueError(f"Stack item tf_modes must be a non-empty list, got {tf_modes!r}")
        axes.append(("tf_mode", list(tf_modes)))
    render_sweep = item.get("render_sweep")
    if render_sweep is not None:
        if not isinstance(render_sweep, dict) or not render_sweep:
            raise ValueError(f"Stack item render_sweep must be a non-empty dict, got {render_sweep!r}")
        for key, values in render_sweep.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"render_sweep.{key} must be a non-empty list, got {values!r}")
            axes.append((f"render.{key}", list(values)))
    return axes


def _suffix_parts_from_combo(combo_items: list[tuple[str, Any]]) -> list[str]:
    parts: list[str] = []
    for key, value in combo_items:
        last_key = key.split(".")[-1]
        normalized_value = _normalize_render_value(last_key, value)
        if last_key in {"cmaps", "cmap"} and isinstance(normalized_value, list):
            value_slug = _slugify("_".join(str(item) for item in normalized_value))
        else:
            value_slug = _slugify(str(normalized_value))
        parts.append(f"{_slugify(last_key)}_{value_slug}")
    return parts


def _expand_stack_item(item: Dict[str, Any]) -> list[Dict[str, Any]]:
    axes = _render_sweep_axes_from_item(item)
    if not axes:
        return [dict(item)]

    base_item = dict(item)
    base_item.pop("cmaps", None)
    base_item.pop("tf_modes", None)
    base_item.pop("render_sweep", None)
    base_name = str(base_item.get("name") or "item")
    expanded_items: list[Dict[str, Any]] = []
    axis_keys = [key for key, _ in axes]
    axis_values = [values for _, values in axes]

    for combo in product(*axis_values):
        expanded = dict(base_item)
        combo_items = list(zip(axis_keys, combo))
        render_overrides = dict(expanded.get("render_overrides") or {})
        for key, value in combo_items:
            if key == "cmap":
                expanded["cmap"] = value
            elif key == "tf_mode":
                expanded["tf_mode"] = value
            elif key.startswith("render."):
                render_overrides[key.split(".", 1)[1]] = _clone_value(value)
            else:
                expanded[key] = _clone_value(value)
        if render_overrides:
            expanded["render_overrides"] = render_overrides
        suffix = "_".join(_suffix_parts_from_combo(combo_items))
        expanded["name"] = f"{base_name}_{suffix}" if suffix else base_name
        expanded_items.append(expanded)
    return expanded_items


def _resolve_stack_items(stack_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    items = stack_data.get("items") or stack_data.get("entries") or stack_data.get("jobs")
    if not isinstance(items, list) or not items:
        raise ValueError("Stack plan must contain a non-empty 'items' list")
    expanded_items: list[Dict[str, Any]] = []
    for item in items:
        expanded_items.extend(_expand_stack_item(dict(item)))
    return expanded_items


def _resolve_stack_template(stack_path: str, stack_data: Dict[str, Any]) -> str:
    template = stack_data.get("template")
    if not template:
        raise ValueError("Stack plan must define 'template'")
    template_path = Path(stack_path).resolve().parent / str(template)
    if not template_path.exists():
        template_path = Path(str(template)).resolve()
    return str(template_path)


def _resolve_stack_output_dir(stack_path: str, stack_data: Dict[str, Any], template_path: str) -> str:
    output_dir = stack_data.get("output_dir") or stack_data.get("output-dir")
    if output_dir:
        if str(output_dir).startswith("/"):
            return str(Path(str(output_dir)).resolve())
        return str((Path(stack_path).resolve().parent / str(output_dir)).resolve())
    return str(Path(template_path).resolve().parent)


def _resolve_stack_dataset_output_root(stack_path: str, stack_data: Dict[str, Any]) -> str:
    dataset_output_root = (
        stack_data.get("dataset_output_root")
        or stack_data.get("dataset-output-root")
        or stack_data.get("outputs_root")
        or stack_data.get("outputs-root")
        or "outputs/generated_stack"
    )
    # Dataset outputs should default to a workspace-level location such as `outputs/...`,
    # not a subdirectory under `configs/`.
    return str(Path(str(dataset_output_root)).resolve())


def _resolve_stack_date_tag(date_tag: str | None = None) -> str:
    if date_tag is not None:
        return date_tag
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_stack_item_output_path(
    template_path: str,
    output_dir: str,
    item: Dict[str, Any],
    index: int,
    date_tag: str,
) -> str:
    template = Path(template_path).resolve()
    item_name = item.get("name") or f"{item['case_start']}_{item['case_end']}_{item['tf_mode']}_{item['cmap']}"
    suffix = _slugify(str(item_name))
    filename = f"{template.stem}_stack_{index:02d}_{suffix}_{date_tag}{template.suffix}"
    return str((Path(output_dir).resolve() / filename))


def _build_item_dataset_output_root(base_output_root: str, item: Dict[str, Any]) -> str:
    item_name = item.get("name")
    if not item_name:
        raise ValueError("Each stack item must define 'name'")
    return str(Path(base_output_root).resolve() / _slugify(str(item_name)))


def generate_config_stack(
    stack_path: str,
    output_dir: str | None = None,
    date_tag: str | None = None,
) -> Dict[str, Any]:
    stack_data = load_config_data(stack_path)
    template_path = _resolve_stack_template(stack_path, stack_data)
    template_data = load_config_data(template_path)
    resolved_output_dir = output_dir or _resolve_stack_output_dir(stack_path, stack_data, template_path)
    resolved_dataset_output_root = _resolve_stack_dataset_output_root(stack_path, stack_data)
    resolved_date_tag = _resolve_stack_date_tag(date_tag)
    items = _resolve_stack_items(stack_data)

    generated_items: list[Dict[str, Any]] = []
    Path(resolved_output_dir).mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(items, start=1):
        missing = [key for key in ("name", "tf_mode", "cmap") if key not in item]
        if missing:
            raise ValueError(f"Stack item #{index} is missing required keys: {', '.join(missing)}")

        generated = build_stack_item_config_data(template_data, item)
        item_dataset_output_root = _build_item_dataset_output_root(resolved_dataset_output_root, item)
        generated.setdefault("batch", {})
        generated["batch"]["output_root"] = item_dataset_output_root
        item_output_path = _build_stack_item_output_path(
            template_path=template_path,
            output_dir=resolved_output_dir,
            item=item,
            index=index,
            date_tag=resolved_date_tag,
        )
        with open(item_output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(generated, f, sort_keys=False, allow_unicode=True)

        generated_items.append(
            {
                "index": index,
                "name": item.get("name"),
                "case_start": str(item["case_start"]) if item.get("case_start") is not None else None,
                "case_end": str(item["case_end"]) if item.get("case_end") is not None else None,
                "tf_mode": resolve_tf_mode(str(item["tf_mode"])),
                "cmap": resolve_cmap_name(str(item["cmap"])),
                "output_root": item_dataset_output_root,
                "config_path": item_output_path,
            }
        )

    return {
        "stack_path": str(Path(stack_path).resolve()),
        "template_path": str(Path(template_path).resolve()),
        "output_dir": str(Path(resolved_output_dir).resolve()),
        "dataset_output_root": str(Path(resolved_dataset_output_root).resolve()),
        "date_tag": resolved_date_tag,
        "items": generated_items,
    }
