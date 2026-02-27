# Vol2PC 开发指南与 TODO

本文档旨在指导如何为 Vol2PC 贡献代码，并列出了当前亟待解决的任务。

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

以下是目前急需完成的功能模块，请师弟优先认领：

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
有时我们不希望改变 Pipeline 中间的坐标系，只想在最后导出 PLY 时强制归一化。

**任务**：
1.  修改 `vol2pc/export/write_ply.py` 中的 `PLYWriter`。
2.  在 `config` 中增加一个参数，例如 `normalize_coords: true`。
3.  如果在写出时该参数为真，则计算点云的重心和尺度，将其平移缩放到单位球或单位立方体内，然后再写入文件。

---

**加油！有问题随时交流。**
