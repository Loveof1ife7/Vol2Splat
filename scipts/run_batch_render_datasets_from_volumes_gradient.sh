#!/bin/bash
# 批量用 Vol2Splat 渲染 datasets_for_volume_3dgs 名单里的体积（启用梯度部分渲染）
# 说明：该入口使用独立配置文件，不影响原有 run_batch_render_datasets_from_volumes.sh。

set -e

if [ -f "/data/wyx/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/data/wyx/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASETS_ROOT="${DATASETS_ROOT:-/root/autodl-tmp/projects/data/datasets_for_volume_3dgs}"
VOLUMES_ROOT="${VOLUMES_ROOT:-/root/autodl-tmp/projects/data/volumes}"
VTI_CACHE_ROOT="${VTI_CACHE_ROOT:-/root/autodl-tmp/projects/data/vti_cache}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/projects/data/datasets_for_volume_3dgs_vol2splat_gradient}"
PVPYTHON="${PVPYTHON:-pvpython}"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/batch_datasets_from_volumes_gradient.yaml}"

echo "=== Vol2Splat Batch Render (datasets list, gradient) ==="
echo "  datasets_root:  $DATASETS_ROOT"
echo "  volumes_root:   $VOLUMES_ROOT"
echo "  vti_cache_root: $VTI_CACHE_ROOT"
echo "  out_root:       $OUT_ROOT"
echo "  pvpython:       $PVPYTHON"
echo "  config:         $CONFIG"
echo ""

python scipts/batch_render_datasets_from_volumes.py \
  --config "$CONFIG" \
  --datasets_root "$DATASETS_ROOT" \
  --volumes_root "$VOLUMES_ROOT" \
  --vti_cache_root "$VTI_CACHE_ROOT" \
  --out_root "$OUT_ROOT" \
  --pvpython "$PVPYTHON"

