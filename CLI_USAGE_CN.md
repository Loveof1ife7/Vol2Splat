# Vol2Splat CLI 使用手册

本文档面向当前仓库里的 `python -m vol2splat.cli` 命令行入口，覆盖常用命令、典型工作流和项目内常见配置方式。

适用入口文件：

- [vol2splat/cli.py](/root/autodl-tmp/projects/Vol2Splat/vol2splat/cli.py:1)

## 1. 总览

CLI 入口：

```bash
python -m vol2splat.cli [全局选项] <command> [命令选项]
```

全局选项：

- `-v`, `--verbose`
  启用 DEBUG 日志。

示例：

```bash
python -m vol2splat.cli -v list
```

## 2. 命令列表

当前 CLI 支持这些命令：

- `run`
- `batch`
- `batch-seg`
- `make-config-stack`
- `list`
- `inspect`
- `test-seg`
- `render-qc`

---

## 3. `run`

用途：

- 跑单个输入文件的完整 pipeline。
- 适合单 case 调试。

命令：

```bash
python -m vol2splat.cli run \
  --config <config.yaml> \
  --input <input_volume> \
  --output <output_path>
```

参数：

- `--config`, `-c`
  配置文件路径，必填。
- `--input`, `-i`
  输入体数据路径，可选；不传时使用配置里的 `io.path`。
- `--output`, `-o`
  输出点云路径，可选；不传时使用配置里的 `export.path`。

示例：

```bash
python -m vol2splat.cli run \
  -c configs/lqb/high_quality/batch_canonical.yaml \
  -i data/vbm_teaser_cases/case001/volume.vti
```

说明：

- `run` 不负责 case discovery。
- 如果你要按目录批量发现输入，优先用 `batch`。

---

## 4. `batch`

用途：

- 对一个数据根目录下的多个 case 批量运行 pipeline。
- 支持 case 发现、过滤、分 batch、按 case 重写 `io.path` / `render.path` / `export.path`。

命令：

```bash
python -m vol2splat.cli batch -c <config.yaml> [更多过滤项]
```

常用参数：

- `--config`, `-c`
  配置文件路径，必填。
- `--raw-root`
  case 根目录。可覆盖 YAML 里的 `batch.raw_root`。
- `--output-root`
  输出根目录。可覆盖 YAML 里的 `batch.output_root`。
- `--case-glob`
  case 目录 glob，例如 `s*` 或 `*`。
- `--case-range`
  case 范围，例如 `s0000~s0100`。
- `--case-start`, `--case-end`
  显式范围边界。
- `--case-ids`
  逗号分隔的 case 子集。
- `--exclude-case-ids`
  逗号分隔的排除列表。
- `--filename`
  若每个 case 目录内输入文件名固定，可显式指定，例如 `canonical.vti`。
- `--continue-on-error/--fail-fast`
  失败后继续或立刻停止。
- `--batch-size`
  一个随机 batch 里多少个 case。
- `--shuffle/--no-shuffle`
  是否打乱。
- `--seed`
  打乱随机种子。
- `--batch-index`
  只执行某个 1-based batch。

示例 1：按 YAML 配置直接跑

```bash
python -m vol2splat.cli batch \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml
```

示例 2：只跑指定 case 子集

```bash
python -m vol2splat.cli batch \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml \
  --case-ids miranda_1024x1024x1024_float32_part_0000,miranda_1024x1024x1024_float32_part_0001 \
  --no-shuffle \
  --batch-size 2
```

示例 3：render-only canonical VTI 数据集

```bash
python -m vol2splat.cli batch \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml
```

说明：

- `batch` 会自动发现 `raw_root/<case_id>/...`。
- 发现后会按 case 重写配置里的输入输出路径。
- 结束后会在 `output_root` 下写：
  `batch_report.json`
  `batch_report.md`

### 4.1 render-only 模式

如果输入本身就是 canonical `.vti`，可以不写 `preprocess`，直接 `io + render`。

典型配置：

- [configs/lqb/batch_canonical_3dgs_sim_render_only.yaml](/root/autodl-tmp/projects/Vol2Splat/configs/lqb/batch_canonical_3dgs_sim_render_only.yaml:1)

当前项目中，这类配置常用于：

- 直接复用 `raw_sim/openscivis_canonical_vti`
- 跳过 canonicalize，直接渲染已有标准化体数据

### 4.2 `batch` 的路径重写语义

`batch` 不是简单把 YAML 原样传下去。它会在每个 case 上展开上下文，并重写：

- `io.path`
- `canonicalize.vti_path`
- `render.path`
- `export.path`

所以：

- YAML 里的路径既可以写死
- 也可以写占位符，例如 `"{batch_output_dir}/{case_id}"`
- 也可以依赖 `batch` 在运行时自动覆盖

---

## 5. `batch-seg`

用途：

- 面向分割后器官体数据的批处理版本。
- 目录结构通常类似：`raw/<case_id>/<seg_subdir>/<organ_file>`

命令：

```bash
python -m vol2splat.cli batch-seg -c <config.yaml> [过滤项]
```

额外常用参数：

- `--seg-subdir`
  器官目录名。
- `--organ-glob`
  器官文件 glob。
- `--organ-ids`
  指定器官子集。
- `--exclude-organ-ids`
  排除器官子集。

示例：

```bash
python -m vol2splat.cli batch-seg \
  -c configs/lqb/batch1_tf2_seg.yaml \
  --raw-root raw/totalseg3d \
  --seg-subdir segmentations
```

---

## 6. `make-config-stack`

用途：

- 从一个 stack plan 自动生成多份 batch YAML。
- 适合做参数 sweep，例如：
  多个 colormap
  多个 tf_mode
  多个 case 范围
- 可选生成后立刻顺序执行。

命令：

```bash
python -m vol2splat.cli make-config-stack \
  --stack <stack.yaml> \
  [--output-dir <dir>] \
  [--run]
```

参数：

- `--stack`, `-s`
  stack plan YAML，必填。
- `--output-dir`
  生成配置文件目录，可选。
- `--run`
  生成后立即逐个执行。

### 6.1 stack plan 结构

最常见字段：

- `template`
  模板 YAML。
- `output_dir`
  生成出来的 config 文件目录。
- `dataset_output_root`
  实际渲染/导出数据集的输出根。
- `items`
  任务列表。

示例：

```yaml
template: batch_canonical_3dgs_sim_render_only.yaml
output_dir: generated_stack
dataset_output_root: outputs/openscivis_canonical_render_only_by_cmap

items:
  - name: miranda_render_only
    tf_mode: linear_gs_sum2
    cmaps:
      - Cool to Warm (Extended)
      - Turbo
      - Jet
```

说明：

- 现在 `items` 里支持直接写 `cmaps: [...]`。
- 现在也支持：
  `tf_modes: [...]`
  `render_sweep: {opaque_unit: [...], opacity_scale: [...], ...}`
- CLI 会自动把一个 item 展开成多份配置，每个 colormap 一份数据集输出。
- 对 `tf_modes` 和 `render_sweep`，会做笛卡尔积展开。
- 这一点适合当前 sim render-only 工作流。

现成示例：

- [configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml](/root/autodl-tmp/projects/Vol2Splat/configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml:1)
- [configs/lqb/batch_canonical_3dgs_sim_render_sweep_stack.yaml](/root/autodl-tmp/projects/Vol2Splat/configs/lqb/batch_canonical_3dgs_sim_render_sweep_stack.yaml:1)

### 6.2 生成示例

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml
```

### 6.3 生成并直接运行

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml \
  --run
```

说明：

- `--run` 会在生成所有展开后的 config 之后，自动按顺序逐个调用 `batch`。
- 如果一个 stack 展开成 `N` 份配置，就会顺序执行 `N` 次 batch。
- 适合参数 sweep 的一键串行执行。

### 6.4 sweep 组合是怎么展开的

当一个 item 里同时存在这些字段时：

- `tf_modes: [...]`
- `cmaps: [...]`
- `render_sweep: {...}`

CLI 会对它们做笛卡尔积展开。

例如：

```yaml
items:
  - name: miranda_render_sweep
    tf_modes:
      - linear_gs_sum2
      - gaussian
    cmaps:
      - Cool to Warm (Extended)
      - Turbo
      - Jet
    render_sweep:
      opaque_unit:
        - 3.0
        - 7.0
      opacity_scale:
        - 0.1
        - 0.3
```

会展开成：

- `2 x 3 x 2 x 2 = 24` 份配置

每一份对应一个唯一组合，例如：

- `tf_mode=linear_gs_sum2, cmap=Cool to Warm (Extended), opaque_unit=3.0, opacity_scale=0.1`
- `tf_mode=gaussian, cmap=Jet, opaque_unit=7.0, opacity_scale=0.3`

输出目录和生成出来的 config 文件名都会带上组合后缀，便于追踪。

### 6.5 如何自动化运行 stack

最常用的两种方式：

只生成配置，不立即跑：

```bash
python -m vol2splat.cli make-config-stack \
  --stack <你的_stack.yaml>
```

生成后立即自动串行运行：

```bash
python -m vol2splat.cli make-config-stack \
  --stack <你的_stack.yaml> \
  --run
```

当前项目里的现成示例：

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_sweep_stack.yaml \
  --run
```

### 6.6 如何保存展开结果摘要

当前 CLI 会把 summary 打印到终端，但不会单独写一个表格文件。

如果你想保留展开结果，可以直接保存终端输出：

```bash
python -m vol2splat.cli make-config-stack \
  --stack <你的_stack.yaml> | tee stack_summary.txt
```

如果还要同时自动运行，也可以：

```bash
python -m vol2splat.cli make-config-stack \
  --stack <你的_stack.yaml> \
  --run | tee stack_run_summary.txt
```

此外，生成后的 config 会写到 `output_dir`，每个文件名本身也带了组合信息。

### 6.7 colormap 名称兼容

当前 stack 解析器支持一些别名，会自动映射到实际 ParaView preset 名：

- `Inferno` -> `Inferno (matplotlib)`
- `Yellow - Grey - Blue` -> `Yellow - Gray - Blue`
- `Bluw - Green - Orange` -> `Blue - Green - Orange`

---

## 7. `list`

用途：

- 查看当前注册的 reader / stage / sampler / renderer / writer。

命令：

```bash
python -m vol2splat.cli list
```

适合用来确认某个插件是否已经注册成功。

---

## 8. `inspect`

用途：

- 快速查看体数据文件的基本信息。

命令：

```bash
python -m vol2splat.cli inspect \
  --input <volume_path> \
  --reader vti
```

输出内容包括：

- shape
- spacing
- origin
- dtype
- value range

示例：

```bash
python -m vol2splat.cli inspect \
  --input raw_sim/openscivis_canonical_vti/miranda_1024x1024x1024_float32_part_0000/miranda_1024x1024x1024_float32_part_0000_canonical.vti \
  --reader vti
```

---

## 9. `test-seg`

用途：

- 不重跑完整 pipeline，直接对已有 render PNG 做 MONAI segmentation QC 测试。

命令方式 1：给 TF 目录

```bash
python -m vol2splat.cli test-seg \
  -c <config.yaml> \
  --tf-dir <render_case/TF01> \
  --split train
```

命令方式 2：直接给图片 glob

```bash
python -m vol2splat.cli test-seg \
  -c <config.yaml> \
  --image-glob "outputs/foo/TF01/train/*.png"
```

参数：

- `--config`, `-c`
  配置文件，要求 `render.qc.segmentation.enabled: true`
- `--tf-dir`
  TF 目录，例如 `outputs/s0001/TF01`
- `--image-glob`
  直接指定 PNG glob
- `--split`
  `train` / `test` / `val`
- `--sample-limit`
  最多评估多少张图
- `--output-json`
  结果 JSON 输出路径

---

## 10. `render-qc`

用途：

- 对已经渲染好的目录重新跑 render QC。
- 不重新 render。

命令：

```bash
python -m vol2splat.cli render-qc \
  -c <config.yaml> \
  --render-dir <case_dir_or_tf_dir>
```

参数：

- `--config`, `-c`
  配置文件，要求有 `render.qc`
- `--render-dir`
  case 目录或单个 TF 目录
- `--report-json`
  可选自定义 QC JSON 输出路径
- `--report-md`
  可选自定义 QC Markdown 输出路径

示例：

```bash
python -m vol2splat.cli render-qc \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml \
  --render-dir outputs/openscivis_canonical_render_only/miranda_1024x1024x1024_float32_part_0000
```

输出内容包括：

- 总 TF 数
- passed / failed 数量
- passed TF 名单
- failed TF 名单
- `render_qc.json`
- `render_qc.md`

---

## 11. 项目内常见工作流

### 11.1 从原始体数据开始：canonicalize + render

适用：

- 输入是原始 `.nii.gz` / `.vti`
- 还没有 canonical VTI

命令：

```bash
python -m vol2splat.cli batch \
  -c configs/lqb/batch_canonical_3dgs_sim.yaml
```

### 11.2 已有 canonical VTI：只 render

适用：

- 输入已经是标准化后的 `.vti`
- 只想复用 render / QC / downstream

命令：

```bash
python -m vol2splat.cli batch \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml
```

### 11.3 一个 case 子集 + 多 colormap 自动拆数据集

适用：

- 例如固定 `miranda part_0000 ~ part_0009`
- 对多个 colormap 各生成一套 render 数据

命令：

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml \
  --run
```

### 11.4 已渲染数据补 canonical.vti 软链接

这个不是 CLI 子命令，但在当前项目工作流里很常用。

脚本：

- [scripts/link_canonical_vti_into_render_dirs.py](/root/autodl-tmp/projects/Vol2Splat/scripts/link_canonical_vti_into_render_dirs.py:1)

示例：

```bash
python scripts/link_canonical_vti_into_render_dirs.py \
  --canonical-root /root/autodl-tmp/projects/Vol2Splat/raw_sim/openscivis_canonical_vti \
  --render-root /root/autodl-tmp/projects/Vol2Splat/outputs/openscivis_canonical_render_only
```

---

## 12. 常见排查

### 12.1 `Batch completed. batches=0 cases=0`

先检查：

- YAML 顶层是不是误写成了 `Cbatch:` 而不是 `batch:`
- `raw_root` 是否指向了真实数据目录
- `case_glob` 是否匹配
- `case_ids` 是否写对

### 12.2 render-only 没有走 preprocess

这是正常行为，只要：

- `io.reader: vti`
- `io.path` 指向真实存在的 `.vti`

pipeline 会直接把输入当成 `canonical_vti_path`。

### 12.3 `make-config-stack` 只想换 colormap，不想改 case 范围

可以。

- 把 case 过滤写在模板 YAML 的 `batch.case_ids`
- stack item 里只写 `name + tf_mode + cmaps`

### 12.4 colormap 名称不生效

优先核对：

- `pvpython_colormaps.txt`
- 生成后的 config 里 `render.cmaps`

若名称里有逗号，建议写成 YAML list，避免被拆错。

---

## 13. 推荐命令速查

查看命令列表：

```bash
python -m vol2splat.cli list
```

单 case 调试：

```bash
python -m vol2splat.cli run -c <config.yaml> -i <input>
```

批量 render-only：

```bash
python -m vol2splat.cli batch -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml
```

生成多 colormap stack：

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml
```

生成并直接跑：

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_only_stack.yaml \
  --run
```

生成 render sweep 并直接跑：

```bash
python -m vol2splat.cli make-config-stack \
  --stack configs/lqb/batch_canonical_3dgs_sim_render_sweep_stack.yaml \
  --run
```

重跑现有 render 的 QC：

```bash
python -m vol2splat.cli render-qc \
  -c configs/lqb/batch_canonical_3dgs_sim_render_only.yaml \
  --render-dir outputs/openscivis_canonical_render_only/miranda_1024x1024x1024_float32_part_0000
```
