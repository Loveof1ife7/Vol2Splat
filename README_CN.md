# Vol2Splat

`Vol2Splat` 现在是一个面向Volmetric data(医学和仿真数据)自动化数据集制作项目

它支持：

- 读取 `nii.gz`、`vti`、`raw`
- 在 `io` 层对超大原始体数据做分块
- 预处理并 canonicalize 成统一体素世界
- 导出 canonical `.vti`
- 用 ParaView `pvpython` 渲染 canonical VTI
- 对 render 结果做自动质检并生成报告
- 基于 `tf_config.json` 烘焙 RGBA Volume再采样PCD
- 批量扫描 `raw/sXXXX` 自动化处理
- 导出给下游 3DGS / NeRF 风格训练使用的点云

## 主流程

```text
io -> preprocess -> canonical.vti -> rendering -> sampling -> export
```

各阶段职责如下：

1. `io`
   只负责读取原始数据。
   保留原始 `spacing / origin / direction`。
   如果输入体数据过大，也在这里触发 tiling，把原始 volume 切成多个 tile 后再交给后续流水线。

2. `preprocess`
   负责把原始体数据转成 canonical voxel world。
   这里做各向同性重采样、可选 pad cube、导出 canonical `.vti`。
   现在默认会在 canonicalize 时先把标量值 min-max 归一化到 `[0,1]`，然后再做 cube padding。
   因此默认 `0` padding 才是合理背景，而 TF 也应基于 canonical volume 的 min/max 来定义。

3. `rendering`
   只读取 canonical `.vti`。
   渲染时使用统一的 target bbox world。
   导出的相机位姿和渲染图像在同一个 world 里。
   如果开启 QC，会在 render 完成后分析各个 `TFxx` 的图像质量，并记录哪些 TF 基本是空背景、过暗或低对比度。
   也可以进一步接入 2D 医学分割模型；如果该模型在某个 TF 的渲染结果上稳定分割失败，就把这个 TF 视为“非有效体素获得了不透明度”。

4. `sampling`
   下游只消费 canonical `.vti` 和 rendering 输出的 `tf_config.json`。
   采样前先消费 `tf_config.json`，把 scalar volume bake 成 RGBA volume。
   采样得到的点云坐标应直接落在 target bbox world。
   如果 rendering 产出多组 TF，比如 `TF01/TF02/...`，pipeline 会把每个 `tf_config.json` 显式传给 sampling，并在对应 TF 文件夹下分别写出结果，例如 `TF01/points.ply`、`TF02/points.ply`。
   如果开启 QC 且设置 `skip_failed_tf: true`，失败的 TF 会自动跳过，不进入 sampling。

5. `export`
   export 只负责写盘，不再承担“把 canonical world 纠正到 render world”的主要职责。
   正常情况下，进入 export 的点云已经在 target bbox world 里。

## 坐标规范

项目里有两层关键坐标：

### 0. 轴顺序约定

这条约定是全项目统一规则：

- `spacing` 是 `(x, y, z)` 顺序
- `origin` 是 `(x, y, z)` 顺序
- `direction` 是基于 `XYZ` 物理轴的 `3x3` 矩阵
- `numpy` 里的标量体数据一律是 `(Z, Y, X)`
- `numpy` 里的 RGBA 体数据一律是 `(4, Z, Y, X)`
- 体素访问写法是 `data[z, y, x]`
- 做坐标变换时，`index_to_world / world_to_index` 的输入输出索引一律用 `(x_idx, y_idx, z_idx)`

也就是说：

- 元数据是 `XYZ` 语义
- numpy volume 是 `ZYX` 语义

这个约定必须在 `io / preprocess / rendering / sampling / export` 全链路保持一致。

### 1. Canonical voxel world

由 `canonicalize` 产生：

- 各向同性
- `origin = (0, 0, 0)`
- `direction = I`
- 标量体默认被归一化到 `[0,1]`
- 体素索引和世界位置保持一致语义

它是项目内部传递体数据的标准格式。

### 2. Render world

由 renderer 从 canonical world 统一缩放得到：

- 按最长边等比例缩放
- 落到固定 target bbox
- 相机位姿也导出在同一坐标系里
- 不仅相机对准 target bbox，真实体渲染本身也发生在 target bbox world 里

下游真正看到的是统一的 render world：

- `images`
- `transforms_*.json`
- `points.ply / points.npz`

另外，render stage 会把多 TF 信息显式暴露给下游，核心信息包括：

- `tf_outputs`: 每个 TF 的结构化信息
- `tf_json`: 对应的 `tf_config.json` 路径
- `tf_dir`: 对应 TF 文件夹路径

sampling 应优先消费这组结构化信息，而不是自己猜目录。

## 项目结构

```text
Vol2Splat/
├── vol2splat/
│   ├── core/          # 核心 pipeline 接口和数据结构
│   ├── io/            # nii.gz / raw / vti reader
│   ├── preprocess/    # normalize / canonicalize / tf
│   ├── common/        # 共享 TF 和几何变换
│   ├── rendering/     # pipeline renderer wrapper + ParaView engine
│   ├── sampling/      # density / opacity / uniform / wavelet
│   └── export/        # ply / npz / vti 导出
├── render/            # 参考渲染引擎源码
├── configs/           # 配置示例
└── tests/
```

## 安装

推荐继续使用你的 `data` 环境：

```bash
conda create -n data python=3.10
conda activate data
pip install -e .
```

如果你想一次装全功能依赖，可以用：

```bash
pip install -e .[full]
```

如果只想补某一类能力：

```bash
pip install -e .[medical]
pip install -e .[render]
pip install -e .[segmentation]
pip install -e .[wavelet]
```

`requirements.txt` 仍然保留，适合你现在这种“固定环境复现”；但日常开发和 API 安装，优先建议直接用 `pyproject.toml` 的 extras。

如果你把它当 Python API 用，安装后可以直接：

```python
from vol2splat import Config, load_config, run_pipeline, register_builtin_plugins

register_builtin_plugins()
cfg = load_config("configs/canonical_3dgs.yaml")
run_pipeline(None, None, cfg)
```

如果要使用 ParaView 渲染，确保系统里可执行 `pvpython`。

## CLI

新的推荐入口：

```bash
vol2splat run -c configs/nii_canonical_pipeline.yaml
```

查看可用插件：

```bash
vol2splat list
python -m vol2splat.cli list
```

查看体数据基本信息：

```bash
vol2splat inspect -i your_case.nii.gz -r nii
python -m vol2splat.cli inspect -i your_case.nii.gz -r nii
```

对已经渲染好的 PNG 单独测试分割 QC：

```bash
vol2splat test-seg -c configs/canonical_3dgs.yaml --tf-dir outputs/s0000/TF01 --split train
python -m vol2splat.cli test_seg -c configs/canonical_3dgs.yaml --tf-dir outputs/s0000/TF01 --split train
```

如果 `render.qc.segmentation` 里配置了 `model_url`，权重会先自动下载到本地，再执行推理。

批量处理一整个 `raw/` 目录：

```bash
vol2splat batch -c configs/batch_canonical_3dgs.yaml --raw-root raw --output-root outputs
python -m vol2splat.cli batch -c configs/batch_canonical_3dgs.yaml --raw-root raw --output-root outputs
```

批量运行结束后，会在 `output-root` 下额外写出：

- `batch_report.json`
- `batch_report.md`

如果 render QC 开启，每个 case 输出目录下还会有：

- `render_qc.json`
- `render_qc.md`

## 示例配置

### 1. `nii.gz -> canonical.vti`

见 [configs/nii_canonical_pipeline.yaml](/data/lqb/datasets/ct-datas/Vol2Splat/configs/nii_canonical_pipeline.yaml)

### 2. `canonical.vti -> render -> opacity sampling -> export`

见 [configs/canonical_3dgs.yaml](/data/lqb/datasets/ct-datas/VolSampler/configs/canonical_3dgs.yaml)

### 3. 批量自动化处理

见 [configs/batch_canonical_3dgs.yaml](/data/lqb/datasets/ct-datas/VolSampler/configs/batch_canonical_3dgs.yaml)

## 开发规范

- reader 只负责读数据，不负责偷偷改坐标系
- canonicalize 属于 preprocess，而不是 reader 内部逻辑
- 超大体数据 tiling 属于 `io`，不属于 `canonicalize`
- canonicalize 默认先做标量 `min-max -> [0,1]` 归一化，再用 `0` 做 cube padding
- 如果确实想改 padding 或关闭这一步，可以显式配置 `pad_value`、`pad_value_mode`、`normalize_scalar`
- 如果原始 volume 非常大，可以在 `io.tiling` 里配置 `tile_size / tile_stride / max_voxels / max_dim`
- renderer 只吃 canonical VTI，不直接处理原始源格式差异
- render QC 是 render 之后、sampling 之前的自动质检层
- render QC 可以只用图像启发式规则，也可以叠加 2D 医学分割模型
- sampler 先消费 rendering 产出的 `tf_config.json`，再基于 RGBA volume 采样
- sampler 的输出点云应直接位于 target bbox world
- export 要和 rendering 使用同一套 render world，但原则上不负责二次坐标纠偏

## 关键文件

- [vol2splat/preprocess/canonicalize.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/preprocess/canonicalize.py)
- [vol2splat/rendering/pv_engine.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/rendering/pv_engine.py)
- [vol2splat/rendering/engine/volume_renderer.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/rendering/engine/volume_renderer.py)
- [vol2splat/sampling/opacity.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/sampling/opacity.py)
- [vol2splat/common/tf.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/common/tf.py)
- [vol2splat/export/write_ply.py](/data/lqb/datasets/ct-datas/Vol2Splat/vol2splat/export/write_ply.py)

## 当前恢复状态

目前已经恢复并可注册的插件包括：

- reader: `nii`, `nii.gz`, `raw`, `vti`
- stage: `canonicalize`, `normalize`, `tf`
- sampler: `opacity`, `uniform`, `wavelet`
- renderer: `pv_engine`
- writer: `npz`, `ply`
