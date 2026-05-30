# 复用已有 TF 扩展 TotalSeg3D 数据集（操作记录）

本文记录如何使用已存在的 TF（`raw/high_quality/medical`）去处理 `raw/totalseg3d` 中更多体积（例如前 20 个），生成新的数据集。

## 1. 关键结论（先看）

- **测试集（`batch_high_quality_medical` 那 7 例）固定不动时**，如何选训练病例、复用 TF、对齐 `canonicalize` 以贴近评测标量行为，见 **第 10 节**。
- 你当前配置 `configs/lqb/high_quality/batch_high_quality_medical.yaml` 已开启：
  - `render.tf_json_root: raw/high_quality/medical`
- 管线会按 `case_id` 自动查找 TF：
  - 优先找 `raw/high_quality/medical/<case_id>/*.json`
  - 若不存在该目录，再找 `raw/high_quality/medical/<case_id>_*.json`
- 因此，**想让某个新病例使用 TF**，就要让 `tf_json_root` 下存在它的 TF 文件（真实拷贝或软链接均可）。

---

## 2. 你的当前状态

- 已有 TF 病例：`s0244, s0652, s0693, s0703, s1009, s1073, s1348`
- 数据源：`raw/totalseg3d`（大量 `sxxxx` 目录）

这意味着：直接跑“前 20 个病例”时，大概率只有极少数能命中同名 TF，其余病例会因无 TF 被 QC 跳过或失败。

---

## 3. 两种可行方案

## 方案 A：仅处理“有同名 TF”的病例（最干净）

适合你只想保证语义一致，不做跨病例复用。

直接在配置中把 `batch.case_ids` 设为已有 TF 的病例即可（例如那 7 个）。

## 方案 B：把已有 TF 复用给新病例（你现在要的）

适合你想扩到前 20 个病例。做法是：

1. 取 `raw/totalseg3d` 的前 20 个 `case_id`
2. 给每个目标 `case_id` 在 `raw/high_quality/medical/<case_id>` 建目录
3. 将某个“源 TF 病例”（如 `s0244`）的 JSON 复用过去（推荐软链接）
4. 跑 batch

---

## 4. 实操步骤（方案 B，前 20 个病例）

以下命令在项目根目录执行：`/root/autodl-tmp/projects/Vol2Splat`

### 4.1 取前 20 个病例 ID

```bash
python - <<'PY'
from pathlib import Path
root = Path("raw/totalseg3d")
cases = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("s")])[:20]
print(",".join(cases))
PY
```

记下输出（示例记为 `CASE_IDS_20`）。

### 4.2 复用已有 TF（推荐软链接）

这里示例把 `s0244` 的 TF 复用给前 20 个病例。

```bash
python - <<'PY'
from pathlib import Path
import os

tf_root = Path("raw/high_quality/medical")
source_case = "s0244"  # 可替换成你想复用的 TF 源病例
src_dir = tf_root / source_case
assert src_dir.is_dir(), f"source tf dir not found: {src_dir}"

cases = sorted([p.name for p in Path("raw/totalseg3d").iterdir() if p.is_dir() and p.name.startswith("s")])[:20]
src_jsons = sorted(src_dir.glob("*.json"))
assert src_jsons, f"no tf json in {src_dir}"

for case in cases:
    dst_dir = tf_root / case
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(src_jsons, start=1):
        dst = dst_dir / f"{case}_{i}.json"
        if dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)

print(f"linked {len(src_jsons)} tf json(s) to each of {len(cases)} case(s)")
PY
```

说明：
- 软链接省空间，且便于后续统一替换 TF。
- 如果你偏好实体文件，把 `os.symlink(...)` 改成 `shutil.copy2(...)`。

### 4.3 新建一个“前 20 病例”配置

从 `configs/lqb/high_quality/batch_high_quality_medical.yaml` 复制一份，例如：

`configs/lqb/high_quality/batch_high_quality_medical_top20.yaml`

只改 `batch.case_ids` 为前 20 的列表，例如：

```yaml
batch:
  raw_root: raw/totalseg3d
  case_ids:
    - s0000
    - s0001
    # ... 直到 20 个
```

其余参数保持不变，尤其保留：

```yaml
render:
  tf_json_root: raw/high_quality/medical
```

### 4.4 跑批处理

```bash
python -m vol2splat.cli batch \
  --config configs/lqb/high_quality/batch_high_quality_medical_top20.yaml \
  --raw-root raw/totalseg3d \
  --output-root outputs/high_quality_medical_top20
```

---

## 5. 结果检查

- 批报告：
  - `outputs/high_quality_medical_top20/batch_report.json`
  - `outputs/high_quality_medical_top20/batch_report.md`
- 单病例输出：
  - `outputs/high_quality_medical_top20/<case_id>/...`

优先看 `batch_report.json` 中每个 case 的：
- `status`
- `failed_tf_names`
- `passed_tf_names`

---

## 6. 建议（避免语义偏移）

- 若跨病例复用同一套 TF（如全部用 `s0244`），数据集风格会更统一，但可能损失个体适配性。
- 更稳妥做法是准备 2~4 组代表性 TF，然后按病例轮换分配（例如 `s0244/s0652/s0703/s1073` 循环）。
- 先小规模（5~10 例）验证，再扩到 20 或更多。

---

## 7. 从头到尾：随机复用 TF（推荐标准流程）

下面是一套你可以重复执行的 SOP，目标是把 `raw/high_quality/medical` 中已有 TF 随机复用到新病例上。

### Step 0：准备

- 进入项目根目录：

```bash
cd /root/autodl-tmp/projects/Vol2Splat
```

- 确认关键路径存在：
  - 输入病例根目录：`raw/totalseg3d`
  - TF 根目录：`raw/high_quality/medical`
  - 基础配置：`configs/lqb/high_quality/batch_high_quality_medical.yaml`

### Step 1：确定要跑的新病例集合（示例：前 20）

```bash
python - <<'PY'
from pathlib import Path
cases = sorted([p.name for p in Path("raw/totalseg3d").iterdir() if p.is_dir() and p.name.startswith("s")])[:20]
print("目标病例数:", len(cases))
print(",".join(cases))
PY
```

### Step 2：收集“可作为 TF 源”的病例

```bash
python - <<'PY'
from pathlib import Path
tf_root = Path("raw/high_quality/medical")
src_cases = sorted([p.name for p in tf_root.iterdir() if p.is_dir() and list(p.glob("*.json"))])
print("可用 TF 源病例数:", len(src_cases))
print(",".join(src_cases))
PY
```

### Step 3：把 TF 随机分配给目标病例（软链接方式）

说明：每个目标病例会随机抽一个 TF 源病例，然后建立 `case_xxx_1.json, case_xxx_2.json...` 的软链接。

```bash
python - <<'PY'
from pathlib import Path
import os
import random

target_cases = sorted([p.name for p in Path("raw/totalseg3d").iterdir() if p.is_dir() and p.name.startswith("s")])[:20]
tf_root = Path("raw/high_quality/medical")
source_cases = sorted([p.name for p in tf_root.iterdir() if p.is_dir() and list((tf_root / p.name).glob("*.json"))])
assert source_cases, "没有可用的 TF 源病例"

seed = 20260502
rng = random.Random(seed)
manifest_lines = [f"seed={seed}"]

for case in target_cases:
    src_case = rng.choice(source_cases)
    src_jsons = sorted((tf_root / src_case).glob("*.json"))
    assert src_jsons, f"TF 源为空: {src_case}"

    dst_dir = tf_root / case
    dst_dir.mkdir(parents=True, exist_ok=True)

    for old in dst_dir.glob("*.json"):
        old.unlink()

    for i, src in enumerate(src_jsons, start=1):
        dst = dst_dir / f"{case}_{i}.json"
        os.symlink(src.resolve(), dst)

    manifest_lines.append(f"{case} <- {src_case} ({len(src_jsons)} files)")

manifest_path = Path("outputs") / "tf_random_reuse_manifest_top20.txt"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
print(f"写入映射清单: {manifest_path}")
print("\n".join(manifest_lines[:8]))
PY
```

### Step 4：创建本次运行专用配置

复制基础配置：

```bash
cp configs/lqb/high_quality/batch_high_quality_medical.yaml \
   configs/lqb/high_quality/batch_high_quality_medical_top20_random_tf.yaml
```

然后编辑 `configs/lqb/high_quality/batch_high_quality_medical_top20_random_tf.yaml`：

- 把 `batch.case_ids` 改成 Step 1 的 20 个病例；
- 保持 `render.tf_json_root: raw/high_quality/medical` 不变。

### Step 5：运行批处理

```bash
python -m vol2splat.cli batch \
  --config configs/lqb/high_quality/batch_high_quality_medical_top20_random_tf.yaml \
  --raw-root raw/totalseg3d \
  --output-root outputs/high_quality_medical_top20_random_tf
```

### Step 6：验收与回看

- 看批报告：
  - `outputs/high_quality_medical_top20_random_tf/batch_report.json`
  - `outputs/high_quality_medical_top20_random_tf/batch_report.md`
- 看 TF 随机映射记录：
  - `outputs/tf_random_reuse_manifest_top20.txt`

如果你对这批结果满意，就固定这个 seed；如果不满意，只改 Step 3 的 `seed` 重做一次随机分配再重跑即可。

---

## 8. SIM TF 跨数据集复用（按前缀分组随机，且仅用 GPU1）

场景：用 `raw/high_quality/sim` 中已有 TF 去处理  
`/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat` 下的其他体数据。

目标约束：
- 随机分配 TF；
- 但每个目标 case 只能从“同前缀”的 `_part_0000` TF 池中抽取。
  - 例：`tacc_turbulence_..._part_0003` 仅从 `tacc_turbulence_..._part_0000` 的 TF 抽；
  - 例：`miranda_..._part_0005` 仅从 `miranda_..._part_0000` 的 TF 抽；
- 运行时固定 `GPU1`，避免占用已有任务。

### 8.1 准备本次输入目录（软链接 canonical.vti）

```bash
cd /root/autodl-tmp/projects/Vol2Splat
mkdir -p raw/high_quality/sim_reuse_from_dataset outputs

python - <<'PY'
from pathlib import Path
import os

src_root = Path('/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat')
dst_root = Path('/root/autodl-tmp/projects/Vol2Splat/raw/high_quality/sim_reuse_from_dataset')
dst_root.mkdir(parents=True, exist_ok=True)

cases = sorted([p for p in src_root.iterdir() if p.is_dir()])
for case_dir in cases:
    case = case_dir.name
    src_vti = case_dir / 'canonical.vti'
    if not src_vti.exists():
        continue
    out_case_dir = dst_root / case
    out_case_dir.mkdir(parents=True, exist_ok=True)
    dst_vti = out_case_dir / 'canonical.vti'
    if dst_vti.exists() or dst_vti.is_symlink():
        dst_vti.unlink()
    os.symlink(src_vti.resolve(), dst_vti)

print(f'prepared {len([p for p in dst_root.iterdir() if p.is_dir()])} cases')
PY
```

### 8.2 生成“按前缀分组”的随机 TF（复制 JSON，非软链接）

说明：目标 case 的 TF 输出到  
`raw/high_quality/sim_reuse_tf/<target_case>/<target_case>_*.json`。

```bash
mkdir -p raw/high_quality/sim_reuse_tf

python - <<'PY'
from pathlib import Path
import random
import shutil
import re

src_tf_root = Path('/root/autodl-tmp/projects/Vol2Splat/raw/high_quality/sim')
target_root = Path('/root/autodl-tmp/projects/Vol2Splat/raw/high_quality/sim_reuse_from_dataset')
out_tf_root = Path('/root/autodl-tmp/projects/Vol2Splat/raw/high_quality/sim_reuse_tf')
manifest = Path('/root/autodl-tmp/projects/Vol2Splat/outputs/sim_reuse_tf_manifest.txt')

seed = 20260502
rng = random.Random(seed)
lines = [f'seed={seed}']

def prefix_key(name: str) -> str:
    m = re.match(r'^(.*)_part_\\d+$', name)
    return m.group(1) if m else name

# 建立可用 TF 池：仅 *_part_0000 或本体同名
pool = {}
for p in src_tf_root.iterdir():
    if not p.is_dir():
        continue
    tf_jsons = sorted(p.glob('*.json'))
    if not tf_jsons:
        continue
    k = prefix_key(p.name)
    if p.name.endswith('_part_0000') or k == p.name:
        pool.setdefault(k, []).append((p.name, tf_jsons))

targets = sorted([p for p in target_root.iterdir() if p.is_dir()])
for case_dir in targets:
    case = case_dir.name
    k = prefix_key(case)
    candidates = pool.get(k, [])
    if not candidates:
        lines.append(f'SKIP {case} :: no tf pool for key={k}')
        continue

    src_case_name, src_jsons = rng.choice(candidates)
    dst_dir = out_tf_root / case
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.glob('*.json'):
        old.unlink()

    for i, src_json in enumerate(src_jsons, start=1):
        dst_json = dst_dir / f'{case}_{i}.json'
        shutil.copy2(src_json, dst_json)
        lines.append(f'{case} <- {src_case_name} :: {src_json.name} -> {dst_json.name}')

manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
print(f'wrote manifest: {manifest}')
PY
```

### 8.3 新建运行配置（复用 sim 配置）

复制：

```bash
cp configs/lqb/high_quality/batch_high_quality_sim.yaml \
   configs/lqb/high_quality/batch_high_quality_sim_reuse_dataset.yaml
```

编辑 `configs/lqb/high_quality/batch_high_quality_sim_reuse_dataset.yaml` 的关键字段：

- `batch.raw_root: raw/high_quality/sim_reuse_from_dataset`
- `batch.output_root: outputs/high_quality_sim_reuse_dataset`
- `io.path: raw/high_quality/sim_reuse_from_dataset/{case_id}/canonical.vti`
- `render.path: outputs/high_quality_sim_reuse_dataset/{case_id}`
- `render.tf_json_root: raw/high_quality/sim_reuse_tf`
- 可删除/清空 `batch.case_ids`（让它按 `case_glob: "*"` 自动发现）

### 8.4 用 GPU1 运行

```bash
CUDA_VISIBLE_DEVICES=1 python -m vol2splat.cli batch \
  --config configs/lqb/high_quality/batch_high_quality_sim_reuse_dataset.yaml \
  --raw-root raw/high_quality/sim_reuse_from_dataset \
  --output-root outputs/high_quality_sim_reuse_dataset
```

### 8.5 验收

- 看复用映射：`outputs/sim_reuse_tf_manifest.txt`
- 看批结果：`outputs/high_quality_sim_reuse_dataset/batch_report.json`

---

## 9. vti_cache 全量处理（排除 sim 已完成）

目标：处理 `data/vti_cache` 下所有体积文件，但排除已在 `raw/high_quality/sim` 中完成过的体积。

实现要点：
- `vti_cache` 中一个目录可能有多个 `.vti`（如 `miranda/tacc` 的 `part_0000~0007`）；
- `batch` 默认“每目录只取一个文件”，因此要先把每个 `.vti` 拆成独立 case；
- case 名使用 `.vti` 文件 stem；
- 排除逻辑：若 stem 已在 `raw/high_quality/sim/<case_id>` 存在，则跳过；
- TF 分配：
  - `*_part_xxxx`：优先同前缀的 `_part_0000` TF 池；
  - 其余 case：从 sim 现有 TF 池随机分配；
- 运行使用 `CUDA_VISIBLE_DEVICES=1`。

本次生成目录：
- 输入：`raw/high_quality/sim_reuse_from_vti_cache`
- TF：`raw/high_quality/sim_reuse_tf_from_vti_cache`
- 配置：`configs/lqb/high_quality/batch_high_quality_sim_vti_cache_exclude_done.yaml`
- 输出：`outputs/high_quality_sim_vti_cache_exclude_done`

---

## 10. 测试集不变：如何造训练集、贴近测试的标量行为、减小域偏移

本节约定：**评测始终使用** `configs/lqb/high_quality/batch_high_quality_medical.yaml` 中的病例（`s0244, s0652, s0693, s0703, s1009, s1073, s1348`）及与之配套的输出目录（例如 `outputs/high_quality_medical`），**不增删测试病例、不改测试列表**。只讨论 **训练数据怎么选、管线怎么对齐**，使训练分布更接近你在测试上关心的行为。

### 10.1 先分清两件事：「TF」与「标量场」

- **医学 TF（JSON）**：描述「标量 → RGBA」的曲线，横轴在工程里通常对应 **canonicalize 之后** 进入渲染/采样的标量（默认约为每例体数据做 `minmax` 后的 `[0,1]`）。
- **标量分布**：同一套 TF 下，不同病人的 **原始 HU**、**FOV、金属伪影** 不同，经 **同一套归一化规则** 后得到的 `[0,1]` 直方图仍可能不同。  
  因此：**只复用测试集里那几套 TF，不等于** 训练时的标量直方图已与测试集完全一致，但通常 **比随便用一套无关 TF 更贴近你真实评测时的上色与透明度逻辑**。

### 10.2「随机 10～20 例 + 只从测试集 TF 池里复用」会不会好一点？

**一般会好一点，尤其在下面两点成立时：**

1. **训练病例与测试病例无交集**（从 `raw/totalseg3d` 抽样时 **显式排除** 上述 7 个 `case_id`，避免信息泄漏）。
2. **训练批次的 preprocess（尤其 `canonicalize`）与跑测试集时完全一致**（同一套 `target_spacing`、`pad_to_cube`、`normalize_scalar`、`normalize_method` 及分位数/固定窗参数）。否则你在对齐 TF，却在标量轴上仍与评测不一致。

**随机取 10～20 优于「按目录名排序的前 20」之处**：排序前 20 对应的是 **低编号子集**，与测试用的中高编号子集 **病人不重叠且统计上也不保证同分布**；在全体约千例中 **均匀随机**（或分层随机，见下）可降低「偶然抽到一整块相似子群体」的风险。

**局限**：TF 与归一化只能对齐 **体渲染管线里的表观与标量轴**；**解剖结构、扫描部位、噪声形态** 仍会带来域偏移，这需要更多样化的训练病例，而不是单靠 TF。

### 10.3 在「不改测试集」前提下，构造更贴近评测的训练集（推荐优先级）

1. **锁 preprocess，与评测 yaml 对齐**  
   训练用独立 yaml 时，把 `preprocess` 段从 `batch_high_quality_medical.yaml` **原样复制**（或 diff 确认一致）。若你希望标量轴 **少受极端体素拉扯**，可在双方 yaml 中同样改为 `canonicalize.normalize_method: percentile`（及相同 `normalize_percentile_low/high`），或经实验后统一为 `fixed_range`（注意：固定 HU 窗时，现有 TF 是否仍合理需做小样本目测/QC）。

2. **训练 TF 只来自测试池**  
   将 `raw/high_quality/medical` 下 **仅那 7 个目录** 作为「TF 源池」，对每个训练 `case_id` 随机或轮换链接其中一套 JSON（与第 7 节随机复用类似，但 `source_cases` 限制为这 7 个）。这样训练时见到的 **曲线形状集合** 与评测一致。

3. **病例抽样：随机 + 排除测试 ID + 固定 seed 可复现**  
   从 `totalseg3d` 全体目录抽样 10～20 个，**排除** 测试 7 ID；`seed` 写入 manifest，便于论文/实验记录。

4. **（可选）按标量统计「贴近测试」再抽样**  
   对测试 7 例与各候选例的 **`ct.nii.gz`（canonical 之前）** 用 **同一随机种子、同一子采样体素数**，在子采样上算 **低维特征**：`mean / std / p1 / p50 / p99`，加上 **固定 HU 轴**（默认约 `-1024`～`3071`）上的 **归一化直方图**（若干 bin）。在 **排除测试 ID** 的候选池中排序取 Top‑K 时，脚本支持两种聚合：  
   - **`mean-prototype`（默认）**：对测试 7 例特征取 **平均** 得到原型，再算距离。若 **测试集内部统计差异很大**，平均向量可能 **不代表任何一个真实测试病例**，选出的训练例会偏向「折中」分布。  
   - **`min-to-test`**：对每个候选，计算到 **每一个** 测试例特征的 L1/L2 距离，取 **最小值** 作为得分（并记录 **nearest_test_id**）。含义是：**只要像测试里的某一个即可**，更适合测试集本身是「多峰」的情形。  
   - **风险**：若用脚本默认 **`--select-mode global`** 只取全局 Top‑K，训练例的 **`nearest_test_id` 可能高度集中在少数几个「在特征空间里特别好碰」的测试锚点上**，其余测试例在训练里几乎没有「近邻」代表。  
   - **缓解**：使用 **`--select-mode balanced-anchors`（仅 `min-to-test`）**：先把每个候选划归其 **argmin 最近测试例** 对应的分桶，再在每桶内按到该锚点的距离排序，对 **7 个测试锚点近似均分名额**（`K=20` 时多为每锚点 2～3 例，余数按顺序多分给前几个锚点）；名额仍不足时用全局距离递补。stderr 会打印本次输出的 **`nearest_test` 计数**，便于检查是否仍偏斜。  
   可选 **`--scale-by-pool`**：各维除以候选池的标准差，减轻量纲差异。  
   **注意**：对齐的是 **原始 HU 上你定义的特征** 的边际相似性，**不是** canonical 后 `[0,1]` 分布；若 preprocess 改了归一化方式，仍要与评测 yaml **一致**。实操见 **第 10.6 节**。

5. **数据量仍偏少时**  
   10～20 例本质是极小训练集，域偏移风险主要来自 **病人多样性不足**；在不动测试集的前提下，尽量 **略增训练例数** 或 **多 seed 多组训练取集成**，比反复调 TF 更有效。

### 10.4 示例：随机 20 例训练 + TF 仅从测试 7 例池中抽取（软链接）

在项目根目录执行；**不会修改** `batch_high_quality_medical.yaml` 本身，只生成训练侧 TF 链接与可选 manifest。

```bash
cd /root/autodl-tmp/projects/Vol2Splat

python - <<'PY'
from pathlib import Path
import os
import random

TEST_CASES = {"s0244", "s0652", "s0693", "s0703", "s1009", "s1073", "s1348"}
tf_root = Path("raw/high_quality/medical")
vol_root = Path("raw/totalseg3d")

all_cases = sorted([p.name for p in vol_root.iterdir() if p.is_dir() and p.name.startswith("s")])
pool = [c for c in all_cases if c not in TEST_CASES]
assert pool, "no training candidates after excluding test ids"

seed = 20260503
rng = random.Random(seed)
k = 20
assert len(pool) >= k, f"need at least {k} train candidates, got {len(pool)}"

target_cases = sorted(rng.sample(pool, k=k))
source_cases = sorted(TEST_CASES)
assert all((tf_root / s).is_dir() and list((tf_root / s).glob("*.json")) for s in source_cases)

manifest = [f"seed={seed}", f"train_n={k}", "tf_source_pool=test_case_ids_only"]

for case in target_cases:
    src_case = rng.choice(source_cases)
    src_jsons = sorted((tf_root / src_case).glob("*.json"))
    dst_dir = tf_root / case
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.glob("*.json"):
        old.unlink()
    for i, src in enumerate(src_jsons, start=1):
        dst = dst_dir / f"{case}_{i}.json"
        os.symlink(src.resolve(), dst)
    manifest.append(f"{case} <- {src_case} ({len(src_jsons)} json)")

out = Path("outputs/tf_reuse_train20_from_test_pool_only.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(manifest) + "\n", encoding="utf-8")
print("train case_ids (copy to yaml batch.case_ids):")
print("\n".join(f"    - {c}" for c in target_cases))
print(f"\nmanifest: {out}")
PY
```

随后：复制一份训练专用 yaml，`batch.case_ids` 填上面打印的 20 个；`preprocess` / `render.tf_json_root` 与评测配置 **保持一致**；`--output-root` 指向单独目录（例如 `outputs/high_quality_medical_train_match_test_tf`），避免覆盖 `outputs/high_quality_medical`。

### 10.5 小结（对照你的三个问题）

| 问题 | 建议 |
|------|------|
| 怎么造训练集使标量行为更贴近测试？ | **同一 preprocess**；可选 **percentile/fixed_range** 与测试 yaml 完全一致；病例上可选 **直方图/统计距离** 抽样。 |
| 随机 10～20 + 复用测试集 TF？ | **通常更好**：TF 族与评测一致 + **排除测试 ID** + **随机优于排序前 20**。 |
| 不动测试集如何减域偏移？ | **对齐管线** > **TF 仅从测试池** > **病例随机/分层** > **略增训练 N**；接受 **解剖域** 无法仅靠 TF 消除。 |

### 10.6 实操：用脚本按 CT 统计选训练病例

仓库脚本：`scripts/select_train_cases_near_test_ct_stats.py`（依赖 `nibabel`：`pip install nibabel` 或 `pip install -e ".[medical]"`）。

在项目根目录执行示例（**全候选池**约千例，首次会较慢；可先加 `--score-pool-size` 做快速试验）：

```bash
cd /root/autodl-tmp/projects/Vol2Splat

# 推荐 A：到「测试特征均值」最近（测试内部差异不大时可用）
python scripts/select_train_cases_near_test_ct_stats.py \
  --totalseg-root raw/totalseg3d \
  --test-case-ids s0244,s0652,s0693,s0703,s1009,s1073,s1348 \
  --aggregate mean-prototype \
  --top-k 20 \
  --seed 0 \
  --n-subsample 200000 \
  --metric l2 \
  --scale-by-pool \
  --json-out outputs/train_select_near_test_stats_mean.json

# 推荐 B：到「某一测试例」最近（测试内部差异大、不想被平均抹平时用）
python scripts/select_train_cases_near_test_ct_stats.py \
  --totalseg-root raw/totalseg3d \
  --test-case-ids s0244,s0652,s0693,s0703,s1009,s1073,s1348 \
  --aggregate min-to-test \
  --select-mode global \
  --top-k 20 \
  --seed 0 \
  --n-subsample 200000 \
  --metric l2 \
  --scale-by-pool \
  --json-out outputs/train_select_near_test_stats_min_global.json

# 推荐 C：min-to-test + 按测试锚点均衡配额（避免 Top-K 全挤在少数测试例上）
python scripts/select_train_cases_near_test_ct_stats.py \
  --totalseg-root raw/totalseg3d \
  --test-case-ids s0244,s0652,s0693,s0703,s1009,s1073,s1348 \
  --aggregate min-to-test \
  --select-mode balanced-anchors \
  --top-k 20 \
  --seed 0 \
  --n-subsample 200000 \
  --metric l2 \
  --scale-by-pool \
  --json-out outputs/train_select_near_test_stats_min_balanced.json

# 快速试验：只在随机 400 个候选上打分（不代替最终全池，仅调参看分布）
python scripts/select_train_cases_near_test_ct_stats.py \
  --totalseg-root raw/totalseg3d \
  --aggregate min-to-test \
  --select-mode balanced-anchors \
  --top-k 20 \
  --seed 1 \
  --n-subsample 100000 \
  --score-pool-size 400 \
  --scale-by-pool
```

标准输出为 **yaml 可直接粘贴的 `case_id` 列表**（带注释距离）；`--json-out` 中含 **`ranked_full_pool`**（全候选按分数排序）、**`output_top_k`**（本次实际输出的 K 例）、`prototype_mean` 与参数。跑 **`min-to-test`** 时 stderr 会打印 **`nearest_test` 计数**，用于检查是否塌缩到个别测试锚点。

选好后：把打印的 `case_id` 写入训练专用 batch yaml；若需 TF，仍按 **10.4** 从测试 7 例池软链；**preprocess 与评测配置保持一致**。

### 10.7 仅渲染 PNG、不采样 / 不导出点云（训练子集试跑）

- 配置：`configs/lqb/high_quality/batch_high_quality_medical_render_only.yaml`（无 `sampling`、`export`；`batch.require_case_ids: true` 防止未指定子集时扫全库）。
- 一键（读 `select` 的 JSON → 软链 TF → batch）：  
  `python scripts/run_train_render_only_from_select_json.py --json-out outputs/train_select_near_test_stats.json --output-root outputs/train_selected_render_only`  
  若 TF 已链好，加 `--skip-link-tf`。
