#!/bin/bash
# 批量用 Vol2Splat 渲染 datasets_for_volume_3dgs 名单里的体积（复用 vti_cache 切块）
#
# 注意：本脚本只提供可复用的运行入口；你说“不要运行”，我不会替你执行。
#
# 可配参数（也可用环境变量覆盖）：
# - DATASETS_ROOT:   选择体积名单（只看目录名）
# - VOLUMES_ROOT:    raw 根目录
# - VTI_CACHE_ROOT:  你当年切块后的 vti 根目录（有 *_part_XXXX.vti 的就直接走 vti）
# - OUT_ROOT:        输出数据集根目录（建议不要覆盖原 datasets_for_volume_3dgs）
# - PVPYTHON:        pvpython 路径
# - CONFIG:          渲染参数配置文件（YAML/JSON）
# - DATASETS:        可选，逗号分隔的数据集名；设置后只跑这些（不必在 DATASETS_ROOT 下建目录）
#                    例：仅 miranda 第一块 + 10 个 TF：
#                    DATASETS="miranda_1024x1024x1024_float32_part_0001" \\
#                    CONFIG="$PROJECT_ROOT/configs/batch_miranda_part0001_gs.yaml" \\
#                    ./scipts/run_batch_render_datasets_from_volumes.sh
#
# 典型输出：
#   $OUT_ROOT/<dataset_or_part>/TF01/train/*.png
#   $OUT_ROOT/<dataset_or_part>/TF01/transforms_train.json
#   $OUT_ROOT/<dataset_or_part>/TF01/tf_config.json
#   $OUT_ROOT/<dataset_or_part>/TF01/metadata.json
#   $OUT_ROOT/<dataset_or_part>/TF01/point_cloud/point_cloud_normalized.ply
# 若在 configs/*.yaml 中设置 sampling.anisotropic_init + export.writer: gs_ply，则上述 PLY
# 为 3D Gaussian Splatting 初始化格式（尺度为 log、opacity 为 logit、rot 为 wxyz）。

set -e

# conda 环境：按需修改（参考 gs-datagen-project/run_vol2gs_data_volumes.sh）
if [ -f "/data/wyx/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/data/wyx/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

# 你可以切到装了 Vol2Splat + ParaView 依赖的环境
# conda activate data

export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASETS_ROOT="${DATASETS_ROOT:-/root/autodl-tmp/projects/data/datasets_for_volume_3dgs}"
VOLUMES_ROOT="${VOLUMES_ROOT:-/root/autodl-tmp/projects/data/volumes}"
VTI_CACHE_ROOT="${VTI_CACHE_ROOT:-/root/autodl-tmp/projects/data/vti_cache}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat}"
PVPYTHON="${PVPYTHON:-pvpython}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/batch_datasets_from_volumes.yaml}"

EXTRA=()
if [ -n "${DATASETS:-}" ]; then
  EXTRA+=(--datasets "$DATASETS")
fi

echo "=== Vol2Splat Batch Render (datasets list) ==="
echo "  datasets_root:  $DATASETS_ROOT"
echo "  volumes_root:   $VOLUMES_ROOT"
echo "  vti_cache_root: $VTI_CACHE_ROOT"
echo "  out_root:       $OUT_ROOT"
echo "  pvpython:       $PVPYTHON"
echo "  config:         $CONFIG"
if [ -n "${DATASETS:-}" ]; then
  echo "  datasets only:  $DATASETS"
fi
echo ""

python scipts/batch_render_datasets_from_volumes.py \
  --config "$CONFIG" \
  --datasets_root "$DATASETS_ROOT" \
  --volumes_root "$VOLUMES_ROOT" \
  --vti_cache_root "$VTI_CACHE_ROOT" \
  --out_root "$OUT_ROOT" \
  --pvpython "$PVPYTHON" \
  "${EXTRA[@]}"

