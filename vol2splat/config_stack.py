from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict

import yaml

from .config import load_config_data


TF_MODE_PRESETS = {
    "single": "linear_gs",
    "double": "linear_gs_sum2",
    "linear_gs": "linear_gs",
    "linear_gs_sum2": "linear_gs_sum2",
}

CMAP_PRESETS = {
    "Viridis": "Viridis (matplotlib)",
    "Turbo": "Turbo",
    "Cool to Warm (Extended)": "Cool to Warm (Extended)",
    "Grayscale": "Grayscale",
    "Inferno": "Inferno (matplotlib)",
    "Yellow - Gray - Blue": "Yellow - Gray - Blue",
    "Black, Blue and White": "Black, Blue and White",
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


def _update_batch_section(batch_cfg: Dict[str, Any], case_start: str, case_end: str) -> None:
    batch_cfg["case_start"] = case_start
    batch_cfg["case_end"] = case_end
    batch_cfg["case_range"] = f"{case_start}~{case_end}"


def _update_render_section(render_cfg: Dict[str, Any], tf_mode: str, cmap_name: str) -> None:
    render_cfg["tf_mode"] = resolve_tf_mode(tf_mode)
    # Store cmaps as a YAML list so names containing commas stay intact.
    render_cfg["cmaps"] = [resolve_cmap_name(cmap_name)]


def build_stack_item_config_data(template_data: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
    config_data = copy.deepcopy(template_data)
    batch_cfg = config_data.setdefault("batch", {})
    render_cfg = config_data.setdefault("render", {})
    _update_batch_section(batch_cfg, case_start=str(item["case_start"]), case_end=str(item["case_end"]))
    _update_render_section(render_cfg, tf_mode=str(item["tf_mode"]), cmap_name=str(item["cmap"]))
    return config_data


def _resolve_stack_items(stack_data: Dict[str, Any]) -> list[Dict[str, Any]]:
    items = stack_data.get("items") or stack_data.get("entries") or stack_data.get("jobs")
    if not isinstance(items, list) or not items:
        raise ValueError("Stack plan must contain a non-empty 'items' list")
    return items


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
        missing = [key for key in ("name", "case_start", "case_end", "tf_mode", "cmap") if key not in item]
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
                "case_start": str(item["case_start"]),
                "case_end": str(item["case_end"]),
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
