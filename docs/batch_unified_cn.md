# 统一批处理参数影响说明（`vol2splat.cli batch`）

本文档聚焦**配置参数对结果的影响**，尤其是体渲染（TF）与采样参数。  
不再展开医学数据集生成流程，仅保留最小入口说明。

---

## 1. 最小入口（只看必需项）

```bash
python -m vol2splat.cli batch \
  --input-mode dataset \
  --dataset-source-root "/root/autodl-tmp/projects/data/vti_cache" \
  --config configs/batch_datasets_from_volumes.yaml \
  --output-root "/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat"
```

入口参数只需理解：

- `--dataset-source-root`：输入体素根目录（可指向 `vti_cache` 或 `raw`）
- `--config`：决定渲染、采样、导出行为的核心配置
- `--output-root`：输出目录

---

## 2. 体渲染参数（`render`）如何影响结果

### 2.1 TF 数量与稳定性

```yaml
render:
  cmaps_random: true
  band_count: 10
```

影响：

- `band_count` 越大，每个体数据会生成更多 TF 分支（`TF01...TFxx`）
- `cmaps_random: true` 会为每个 case 随机采样 colormap
- 批处理中配合固定 `batch.seed` 时，TF 随机结果可复现

如需固定 TF 集合：

```yaml
render:
  cmaps: ["Viridis (matplotlib)", "Inferno (matplotlib)"]
  band_count: 2
```

### 2.2 `tf_mode` 与视觉风格

常见模式：

- `linear_vol2splat`
- `linear_gs`
- `linear_gs_sum2`

影响（经验）：

- `linear_vol2splat`：灰度到透明度映射更直接，结果更“保守”
- `linear_gs` / `linear_gs_sum2`：更偏向 GS 训练可用的颜色-透明度分布

### 2.3 渲染 QC 对后续采样的影响

当配置启用 QC 且 `skip_failed_tf: true` 时：

- 失败 TF 不进入采样阶段
- 空 TF 目录会在 batch 收尾清理
- 最终 `_batch_done.json` 只记录有效 TF

---

## 3. 采样参数（`sampling`）如何影响点云

### 3.1 采样器选择

```yaml
sampling:
  name: opacity
```

可用采样器（以插件注册为准）：

- `opacity`：基于 TF alpha 的主力采样
- `uniform`：几何上更均匀
- `wavelet`：频域结构驱动

### 3.2 `opacity` 关键参数

```yaml
sampling:
  name: opacity
  n_points: 200000
  tf_backend: torch
  tf_device: cuda
  uniform_tf_filter: true
  uniform_stride: auto
  alpha_threshold: 0.0
  jitter: true
  anisotropic_init: true
```

参数影响：

- `n_points`：目标点数上限，直接影响密度与体积
- `uniform_tf_filter`：先做规则体素候选，再按 alpha 过滤；点云更均匀但可能更稀
- `uniform_stride`：
  - 固定整数：控制候选网格稀疏度
  - `auto`：根据 `n_points` 与 alpha 保留率自动估计步长
- `alpha_threshold`：阈值越高，低透明区域点越少
- `jitter`（即 use_jitter 语义）：给采样坐标加小扰动，降低网格感
- `anisotropic_init`：为 GS 写入初始各向异性尺度与旋转属性

### 3.3 `opacity` 的代码分层建议

为避免主流程膨胀，推荐保持如下分层（已按该方向实现）：

- `opacity.py`：只保留主流程编排（读取配置、组织调用、组装输出）
- `sampling/low_level`：封装细节计算
  - `uniform_tf_filter / auto_stride`
  - `jitter`
  - `anisotropic_init`

这与 `wavelet` 的“主流程 + low_level”组织方式一致，更利于维护和替换策略。

---

## 4. 导出参数（`export`）如何影响产物

### 4.1 导出器选择

- `export.writer: ply`：通用点云（便于常规可视化与检查）
- `export.writer: gs_ply`：面向 3D Gaussian Splatting 训练

### 4.2 `gs_ply` 常用参数

```yaml
export:
  writer: gs_ply
  params:
    sh_degree: 3
    negate_yz_axes: true
    default_linear_scale: 0.01
```

影响：

- `sh_degree`：控制颜色球谐阶数，影响表达能力与存储规模
- `negate_yz_axes`：坐标系约定修正，关系到训练时方向是否一致
- `default_linear_scale`：缺省高斯尺度，影响初始 splat 大小

---

## 5. 建议的调参顺序（体渲染优先）

推荐按下面顺序调参，避免同时改太多项：

1. 先定 `render`：`tf_mode`、`cmaps_random/cmaps`、`band_count`
2. 再定 `sampling`：`n_points`、`uniform_tf_filter`、`alpha_threshold`
3. 最后定 `export`：`writer` 与 `gs_ply` 参数

这样可以把“可见结构变化”与“点云采样变化”分开观察，定位问题更快。
