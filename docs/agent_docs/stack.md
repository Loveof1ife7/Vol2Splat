# Config Notes

## Colormap presets
Stack config generation currently supports these colormap choices:

- `Viridis` -> `Viridis (matplotlib)`
- `Turbo` -> `Turbo`
- `Cool to Warm (Extended)` -> `Cool to Warm (Extended)`
- `Grayscale` -> `Grayscale`
- `Inferno` -> `Inferno (matplotlib)`
- `Yellow - Gray - Blue` -> `Yellow - Gray - Blue`
- `Black, Blue and White` -> `Black, Blue and White`

The template config is [configs/lqb/batch1_tf1.yaml](/root/autodl-tmp/projects/Vol2Splat/configs/lqb/batch1_tf1.yaml:1).

## TF mode

- single peak: `linear_gs`
- double peak: `linear_gs_sum2`

## Stack plan mode
For large-scale dataset generation, you can describe multiple case ranges in one stack plan.

Example stack plan: [configs/lqb/batch1_stack.yaml](/root/autodl-tmp/projects/Vol2Splat/configs/lqb/batch1_stack.yaml:1)

```yaml
template: batch1_tf1.yaml
output_dir: generated_stack
dataset_output_root: outputs/generated_stack

items:
  - name: range_0000_0025_ygb_single
    case_start: s0000
    case_end: s0025
    tf_mode: linear_gs
    cmap: Yellow - Gray - Blue

  - name: range_0026_0050_inferno_single
    case_start: s0026
    case_end: s0050
    tf_mode: linear_gs
    cmap: Inferno
```

Generate all configs:

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch1_stack.yaml
```

Generate and run them sequentially:

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch1_stack.yaml \
  --run
```

This workflow does not prompt for confirmation. Generation and execution are controlled explicitly:

- omit `--run`: only generate configs
- add `--run`: generate configs and execute them sequentially

Each stack item generates one YAML and can use a different:

- `case_start`
- `case_end`
- `tf_mode`
- `cmap`

Each generated config also gets its own dataset output root based on `items[].name`, for example:

```text
outputs/generated_stack/range_0000_0025_ygb_single
```

This keeps outputs separated by the stack item name only.

## Cmap storage format
Configs now store `render.cmaps` as a YAML list, for example:

```yaml
render:
  cmaps:
    - Black, Blue and White
```

This avoids breaking preset names that contain commas.
