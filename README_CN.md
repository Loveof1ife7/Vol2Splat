# VolSampler: Forging Sparse Representations from Volume

体数据（Volume）转点云（Point Cloud）工具。它主要用于将科学可视化的体数据（如 `.vti` 格式）通过不同的采样策略转换为点云，方便后续进行 3D Gaussian Splatting (3DGS) 训练或其他点云处理任务。

## 核心功能

这个项目的核心在于**模块化管道 (Pipeline)**：
`Reader` -> `Preprocess` -> `Sampler` -> `Writer`
## 项目结构说明

本项目采用分层架构，主要模块如下：

```text
VolSampler/
├── vol2pc/
│   ├── core/           # [核心] 定义基础数据结构 (Volume, PointCloud) 和 Pipeline 接口
│   ├── io/             # [IO模块] 负责文件读取 (目前支持 .vti)
│   ├── preprocess/     # [预处理] 数据归一化 (normalize)、传输函数映射 (tf)
│   │   ├── tf.py       # 传输函数具体实现
│   │   └── low_level/  # 底层加速算子 (CUDA/Torch 优化实现)
│   ├── sampling/       # [采样器] 核心采样算法
│   │   ├── wavelet.py  # 小波采样器 (WaveletSampler)
│   │   └── low_level/  # 小波变换底层实现 (dwt3, sparsify)
│   ├── export/         # [导出] 负责结果写出 (目前支持 .ply)
│   ├── cli.py          # [CLI] 命令行入口，负责解析参数和运行 Pipeline
│   ├── config.py       # 配置加载与解析
│   └── registry.py     # 插件注册机制
├── configs/            # 配置文件示例 (yaml)
├── datasets/           # 示例数据集
├── examples/           # 演示脚本
└── tests/              # 单元测试
```

可以自由组合不同的模块来处理数据。目前最强力的功能是基于**小波变换 (Wavelet Transform)** 的重要性采样，能够根据体数据的特征（边缘、纹理）自动分配采样点

## 环境配置

推荐使用 `conda` 创建环境：

```bash
conda create -n vol2pc python=3.9
conda activate vol2pc
pip install -r requirements.txt
# 或者直接安装本项目
pip install -e .
```

依赖库主要包括：`numpy`, `torch`, `ptwt` (PyTorch Wavelet), `typer` (CLI), `pyyaml`.

## 快速上手

### 1. 命令行运行 (CLI)

这是最常用的方式。你需要准备一个配置文件 (`.yaml`)。请参考 `configs/`

```bash
python -m vol2pc.cli run -c configs/your_config.yaml
```

如果需要覆盖输入输出路径：

```bash
python -m vol2pc.cli run -i datasets/data.vti -c configs/your_config.yaml -o output.ply
```

### 2. 配置文件详解

配置文件是核心。下面是两个典型的场景：

#### 场景 A：基于频率分布对密度体进行采样 (Density Only), 请参考 `configs/wl_rgba.yaml`

如果你只需要根据密度的变化来采样点（不带颜色，或者只看几何结构），用这个配置。注意 **不要** 加 `normalize`，除非你确定 TF 需要归一化后的输入。

```yaml
io:
  reader: vti

preprocess:
  # 原始物理数值 -> [0,1]
  - name: normalize 
    method: minmax

sampling:
  name: wavelet
  n_points: 50000    # 采样点数
  wavelet: haar      # 小波基，目前 haar 最快最稳
  level: 3           # 小波分解层数，3 层通常够用了
  device: cuda       # 必须用 cuda，CPU 跑不动

export:
  writer: ply
```

![alt text](figures/w_supernova_rgba.png)

#### 场景 B：基于频率分布对RGBA体进行采样 (TF) `configs/wl_density.yaml`

这是做可视化的重点。流程是：`密度 -> TF -> RGBA -> 小波采样`。
这样采样出来的点云不仅有位置，还有 TF 映射后的颜色和透明度。

**注意：** 如果你的 TF 定义域是原始物理值（比如 0~100），千万**不要**开 `normalize` 预处理，否则数据会被压到 0~1，导致 TF 映射失效（全是透明背景）。

```yaml
io:
  reader: vti

preprocess:
  # 1. 归一化 (可选，取决于你的 TF 是针对原始值还是 0-1 设计的)
  # - name: normalize
  #   method: minmax
  
  # 2. 应用传输函数
  - name: tf
    tf_json: "path/to/your_tf.json"  # 你的 TF 配置文件
    device: cuda

sampling:
  name: wavelet
  n_points: 100000
  level: 3
  device: cuda

export:
  writer: ply
  path: results/your.ply
```
![alt text](figures/w_supernova_density.png)

#### 场景C：基于梯度分布对密度体进行采样 `configs/g_density.yaml`
```
  io:
    reader: vti
    path: datasets/supernova_432x432x432_float32x.vti

  preprocess:
    - name: normalize
      method: minmax

  sampling:
    name: gradient
    n_points: 100000

  export:
    writer: ply
    path: results/supernova_density_gradient.ply

```

![alt text](figures/g_supernova_density.png)

### TF JSON 格式说明

`tf_json` 需要符合以下格式（控制点列表）：

```json
{
  "control_points": [
    [ -12.0,  0.0, 0.0, 0.0, 0.0 ],
    [ 0.5,    1.0, 0.0, 0.0, 0.1 ],
    ...
  ]
}
```
每一行是 `[Scalar_Value, R, G, B, Alpha]`。

## 如何开发新插件

Vol2PC 采用了基于注册表的插件系统。所有的扩展功能（Reader, Stage, Sampler, Writer）都应通过继承基类并注册来实现。



### 1. 继承基类

根据你要开发的功能，从 `vol2pc.core.pipeline` 导入相应的基类：

*   `Reader`: 读取文件格式
*   `Stage`: 预处理步骤（如归一化、滤波、TF映射）
*   `Sampler`: 采样策略（如随机、梯度、小波）
*   `Writer`: 导出格式

### 2. 实现核心逻辑

以开发一个新的 `MySampler` 为例：

```python
from typing import Dict, Any
from vol2pc.core.types import Volume, PointCloud
from vol2pc.core.pipeline import Sampler
from vol2pc.registry import register_sampler

class MySampler(Sampler):
    def sample(self, vol: Volume, cfg: Dict[str, Any]) -> PointCloud:
        # 获取配置参数
        n_points = cfg.get("n_points", 1000)
        
        # 实现采样逻辑...
        # 注意：vol.data 可能是 (Z,Y,X) 单通道，也可能是 (4,Z,Y,X) RGBA
        
        # 返回 PointCloud 对象
        return PointCloud(xyz=xyz_coords, attrs={"color": colors})

# 3. 注册插件
# 这一步非常重要！只有注册了，CLI 才能通过名字找到它。
register_sampler("my_sampler", MySampler)
```

### 3. 启用插件

确保你的新文件被导入。通常在 `vol2pc/<subpackage>/__init__.py` 中添加 `from .my_sampler import MySampler` 即可，系统会自动扫描并注册。

---

## TODO List

### TODO 1: 梯度采样器的 RGBA 版本支持

**现状**：
目前的 `GradientSampler` (`vol2pc/sampling/gradient.py`) 仅支持单通道密度体采样。如果输入经过了 `tf` 预处理变成了 RGBA (4通道) 数据，它会报错或者行为不正确。

**任务**：
1.  修改 `GradientSampler.sample` 方法。
2.  增加对 `vol.data` 维度的检查：如果是 4D (C, Z, Y, X)，需要先计算梯度的模长（可以对 RGBA 的 luminance 或者 alpha 通道求梯度，或者综合各通道梯度）。
3.  在采样得到坐标后，如果输入是 RGBA，需要像 `WaveletSampler` 那样使用 `grid_sample` 插值出对应点的颜色值，并存入 `PointCloud.attrs`。

### TODO 2: 预处理中的坐标归一化 (Coordinate Normalization Stage)

**现状**：
目前我们输出的点云坐标通常是基于体素索引（Index Space）转换来的物理坐标（World Space）。但在某些渲染引擎（如 3DGS 查看器）中，通常要求坐标在单位立方体 `[-1, 1]` 或 `[0, 1]` 之间。

**任务**：
1.  在 `vol2pc/preprocess/` 下新建一个 `coord.py`。
2.  实现 `CoordNormalizeStage`。
3.  逻辑：读取 `vol.origin` 和 `vol.spacing` 以及 `vol.shape`，计算出包围盒 (Bounding Box)，然后修改 `vol.origin` 和 `vol.spacing`，使得整个体数据被缩放到指定的范围内（如 `[-1, 1]`）。
4.  注意：这个 Stage 改变的是元数据（Metadata），而不一定是体素数据本身（Data）。但如果后续流程依赖元数据转坐标，结果就会自动归一化。

### TODO 3: Export 阶段的坐标归一化

**现状**：
希望能在最后导出 PLY 时强制归一化。配适下游FF 

**任务**：
1.  修改 `vol2pc/export/write_ply.py` 中的 `PLYWriter`。
2.  在 `config` 中增加一个参数，例如 `normalize_coords: true`。
3.  如果在写出时该参数为真，则计算点云的重心和尺度，将其平移缩放到单位球或单位立方体内，然后再写入文件。




