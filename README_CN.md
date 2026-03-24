# Vol2Splat

`Vol2Splat` 现在是一个面向医学体数据和科学可视化体数据的完整流水线项目。

它支持：

- 读取 `nii.gz`、`vti`、`raw`
- 预处理并 canonicalize 成统一体素世界
- 导出 canonical `.vti`
- 用 ParaView `pvpython` 渲染 canonical VTI
- 基于 `tf_config.json` 烘焙 RGBA 再采样
- 导出给下游 3DGS / NeRF 风格训练使用的点云

## 主流程

```text
io -> preprocess -> canonical.vti -> rendering -> sampling -> export
```

各阶段职责如下：

1. `io`
   只负责读取原始数据。
   保留原始 `spacing / origin / direction`。

2. `preprocess`
   负责把原始体数据转成 canonical voxel world。
   这里做各向同性重采样、可选 pad cube、导出 canonical `.vti`。

3. `rendering`
   只读取 canonical `.vti`。
   渲染时使用统一的 target bbox world。
   导出的相机位姿和渲染图像在同一个 world 里。

4. `sampling`
   下游只消费 canonical `.vti` 和 rendering 输出的 `tf_config.json`。
   采样前先消费 `tf_config.json`，把 scalar volume bake 成 RGBA volume。
   采样得到的点云坐标应直接落在 target bbox world。

5. `export`
   export 只负责写盘，不再承担“把 canonical world 纠正到 render world”的主要职责。
   正常情况下，进入 export 的点云已经在 target bbox world 里。

## 坐标规范

项目里有两层关键坐标：

### 1. Canonical voxel world

由 `canonicalize` 产生：

- 各向同性
- `origin = (0, 0, 0)`
- `direction = I`
- 体素索引和体素世界位置保持一致语义

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
pip install -r requirements.txt
pip install -e .
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
```

查看体数据基本信息：

```bash
vol2splat inspect -i your_case.nii.gz -r nii
```

## 示例配置

### 1. `nii.gz -> canonical.vti`

见 [configs/nii_canonical_pipeline.yaml](/data/lqb/datasets/ct-datas/Vol2Splat/configs/nii_canonical_pipeline.yaml)

### 2. `canonical.vti -> render -> opacity sampling -> export`

见 [configs/canonical_3dgs.yaml](/data/lqb/datasets/ct-datas/Vol2Splat/configs/canonical_3dgs.yaml)

## 开发规范

- reader 只负责读数据，不负责偷偷改坐标系
- canonicalize 属于 preprocess，而不是 reader 内部逻辑
- renderer 只吃 canonical VTI，不直接处理原始源格式差异
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
- sampler: `density`, `opacity`, `uniform`, `wavelet`
- renderer: `pv_engine`
- writer: `npz`, `ply`
