# 统一批处理文档（`vol2splat.cli batch`）

本文档说明统一后的批处理入口：

```bash
python -m vol2splat.cli batch ...
```

它覆盖两类任务：

- `case` 模式：面向 `raw/sXXXX` 的病例批处理
- `dataset` 模式：面向 `datasets + vti_cache/volumes` 的数据集批处理

---

## 1. 数据集输入输出参数和数据结构

### 1.1 `case` 模式（医学病例目录）

典型命令：

```bash
python -m vol2splat.cli batch \
  --input-mode case \
  --config configs/batch_medical_20.yaml \
  --raw-root /root/autodl-tmp/projects/Vol2Splat/raw \
  --output-root /root/autodl-tmp/projects/data/medical_dataset_vol2splat_20 \
  --batch-size 20 \
  --batch-index 1 \
  --no-shuffle
```

输入发现规则：

- 扫描 `--raw-root` 下匹配 `--case-glob` 的目录（默认 `s*`）
- 按配置 `io.reader` 选择候选文件（`nii` 时为 `*.nii.gz/*.nii`）
- 同目录内优先使用 `ct.nii.gz`、`ct.nii`、`volume.vti`

输出结构（启用批次）：

```text
<output-root>/
  batch_0001/
    s0001/
      TF01/... TF02/...
      render_qc.json / render_qc.md
    s0002/
    ...
  batch_report.json
  batch_report.md
```

### 1.2 `dataset` 模式（数据集列表 + vti cache）

典型命令：

```bash
python -m vol2splat.cli batch \
  --input-mode dataset \
  --config configs/batch_datasets_from_volumes.yaml \
  --dataset-source-root "/root/autodl-tmp/projects/data/vti_cache" \
  --datasets "$DATASETS" \
  --output-root "/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat" \
  --max-vti-parts 2 \
  --skip-existing
```

输入发现规则：

- 若给 `--datasets`：按逗号分隔名称处理
- 否则：扫描 `--datasets-root` 一级子目录名作为数据集名
- 若给 `--dataset-source-root`，优先使用该目录作为唯一输入源：
  - 目录下存在 `*.vti` 子目录时，按 VTI 模式处理
  - 否则递归扫描该目录下 `*.raw` 文件并按 RAW 模式处理
- 若给 `--data-root`，会自动映射：
  - `--datasets-root = {data-root}/datasets_for_volume_3dgs`
  - `--vti-cache-root = {data-root}/vti_cache`
  - `--volumes-root = {data-root}/volumes`
- 每个数据集先查 `--vti-cache-root/<dataset_name>/*.vti`
- 若名称带 `_part_XXXX`，会按 part 精确匹配
- 若找不到 VTI，默认回退到 `--volumes-root` 递归找 `*.raw`
- 若指定 `--dataset-vti-only`，则不回退 `raw`，该数据集直接跳过
- `raw` 数据集名必须可解析 `name_XxYxZ_dtype[_part_0000]`

输出结构：

- 若输入来自 VTI part：输出目录名用 `part stem`（例如 `xxx_part_0000`）
- 若输入来自 RAW：输出目录名用数据集名
- 每个 case 目录下增加 `_batch_done.json`

### 1.3 与旧脚本等价的关键参数

- `--dataset-source-root`：单路径输入模式（推荐，自动判断 VTI/RAW）
- `--data-root`：统一指定 dataset 输入根目录（推荐）
- `--skip-existing`：跳过完整输出（检查 `_batch_done.json` 或 TF 目录完整性）
- `--max-vti-parts N`：每个数据集最多处理前 N 个 VTI part
- `--dataset-vti-only`：只允许从 `vti_cache` 取输入，禁用 `raw` 回退
- `--batch-size / --batch-index / --shuffle / --seed`：统一批调度策略

### 1.4 常见场景：删除某个 VTI 后不希望被重新生成

例如你删除了：

- `/root/autodl-tmp/projects/data/vti_cache/blunt_fin_256x128x64_uint8/...`

但 `volumes` 里仍有同名 raw，默认会触发 raw 回退并重新生成。  
这时请在命令里加 `--dataset-vti-only`，即可避免该数据集被生成。

如果你改用单路径输入：

- `--dataset-source-root /root/autodl-tmp/projects/data/vti_cache`

则不会访问 `volumes`，删除掉的 VTI 也不会因 raw 回退而被重新生成。

---

## 2. 渲染参数（重点：随机 TF 与不同 TF 类型）

### 2.1 随机 TF 选择（推荐方式）

在配置中使用：

```yaml
render:
  cmaps_random: true
  band_count: 10
```

行为说明：

- 每个 case 自动生成 `band_count` 个 colormap
- 每个 case 的随机结果可复现（由 `batch.seed + case_id` 派生）
- 渲染产生 `TF01 ... TFxx`

如需固定 TF，可直接写：

```yaml
render:
  cmaps: ["Viridis (matplotlib)", "Inferno (matplotlib)", "..."]
  band_count: 10
```

### 2.2 `tf_mode` 选择建议

- `linear_vol2splat`：更贴合当前医学流程默认
- `linear_gs` / `linear_gs_sum2`：更偏 3DGS 数据风格

建议配套关系：

- 医学病例渲染优先从 `linear_vol2splat + opacity_scale 0.3~0.6` 起步
- 3DGS 训练集优先 `linear_gs_sum2`，并结合采样过滤参数

### 2.3 多 TF 的筛选与保留

- 若 render QC 开启且 `skip_failed_tf: true`，失败 TF 不进入采样
- batch 收尾阶段会清理“未采样到的空 TF 目录”
- 最终只保留有效 TF，并写入 `_batch_done.json`

---

## 3. 点云生成方式和设置

### 3.1 采样器类型

项目内可用采样器（以插件注册为准）：

- `opacity`：当前批处理主力方式（渲染 TF 驱动）
- `uniform`：更均匀的几何采样
- `wavelet`：频域/结构敏感采样

批量场景推荐先用：

```yaml
sampling:
  name: opacity
```

### 3.2 `opacity` 常用参数

```yaml
sampling:
  name: opacity
  n_points: 200000
  tf_backend: torch
  tf_device: cuda
  uniform_tf_filter: false
  alpha_threshold: 0.0
  jitter: true
```

说明：

- `uniform_tf_filter: true` 会先做均匀候选再用 alpha 过滤，可能更“稀”
- `alpha_threshold` 越高，保留点越少
- `jitter` 会增加采样位置随机扰动

### 3.3 导出器类型

- `export.writer: ply`：通用点云导出
- `export.writer: gs_ply`：面向 3D Gaussian Splatting 的导出格式

3DGS 训练建议（示例）：

```yaml
export:
  writer: gs_ply
  params:
    sh_degree: 3
    negate_yz_axes: true
    default_linear_scale: 0.01
```

---

## 4. 最小迁移建议

从旧 datasets 批处理脚本迁移时，优先保留：

1. `--input-mode dataset`
2. `--datasets / --datasets-root / --vti-cache-root / --volumes-root`
3. `--max-vti-parts`
4. `--skip-existing`
5. 配置文件中的 `sampling` 与 `export` 段

这样可以在不改核心 pipeline 的前提下，保持与旧流程近似的实际行为。
