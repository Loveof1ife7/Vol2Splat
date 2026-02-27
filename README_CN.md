# VolSampler: 体数据转点云采样工具

体数据（Volume）转点云（Point Cloud）工具。它主要用于将科学可视化的体数据（如 `.vti` 格式）通过不同的采样策略转换为点云，方便后续进行 3D Gaussian Splatting (3DGS) 训练或其他点云处理任务。

## 核心功能

这个项目的核心在于**模块化管道 (Pipeline)**：
`Reader` -> `Preprocess` -> `Sampler` -> `Writer`

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

这是最常用的方式。你需要准备一个配置文件 (`.yaml`)。

```bash
python -m vol2pc.cli run -c configs/your_config.yaml
```

如果需要覆盖输入输出路径：

```bash
python -m vol2pc.cli run -i datasets/data.vti -c configs/your_config.yaml -o output.ply
```

### 2. 配置文件详解

配置文件是核心。下面是两个典型的场景：

#### 场景 A：直接对密度体进行采样 (Density Only)

如果你只需要根据密度的变化来采样点（不带颜色，或者只看几何结构），用这个配置。注意 **不要** 加 `normalize`，除非你确定 TF 需要归一化后的输入。

```yaml
io:
  reader: vti

preprocess:
  # 通常不需要 normalize，保留原始物理数值
  # - name: normalize 
  #   method: minmax

sampling:
  name: wavelet
  n_points: 50000    # 采样点数
  wavelet: haar      # 小波基，目前 haar 最快最稳
  level: 3           # 小波分解层数，3 层通常够用了
  device: cuda       # 必须用 cuda，CPU 跑不动

export:
  writer: ply
```

#### 场景 B：应用传输函数 (Transfer Function) 生成带色点云 (RGBA)

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
```

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

## 常见问题 (FAQ)

1.  **输出的点云全是黑的或者空的？**
    *   检查 `preprocess` 里是不是加了 `normalize`。如果你的 TF 是针对原始数据范围设计的（比如温度 2000~5000），而你做了 minmax 归一化（变到 0~1），TF 查表就会全落在第一个控制点（通常是透明/黑色）。**把 `normalize` 去掉试试。**
    *   检查 TF JSON 的路径是否正确。

2.  **显存爆了 (OOM)？**
    *   小波变换需要一定的显存。如果数据很大（比如 512^3），代码里已经做了 `chunking`（分块处理），但如果还是爆，尝试减小 `chunk_size` 或者把数据缩小一点。

3.  **如何调试？**
    *   可以使用 VS Code 的 Debug 模式，`.vscode/launch.json` 我已经配好了，直接选 `Vol2PC: Run Pipeline` 就能跑。

## 代码结构

*   `vol2pc/sampling/wavelet.py`: 小波采样的核心逻辑。
*   `vol2pc/preprocess/tf.py`: 传输函数映射逻辑。
*   `vol2pc/preprocess/low_level/`: 底层加速算子。

有问题随时问我。
