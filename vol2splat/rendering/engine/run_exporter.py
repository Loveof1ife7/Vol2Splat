import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def build_parser():
    ap = argparse.ArgumentParser(description='Run integrated ParaView exporter for canonical VTI input')
    ap.add_argument('--vti', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--num', type=int, default=64)
    ap.add_argument('--w', type=int, default=800)
    ap.add_argument('--h', type=int, default=800)
    ap.add_argument('--vfov', type=float, default=45.0)
    ap.add_argument('--radius_scale', '--radius-scale', dest='radius_scale', type=float, default=0.8)
    ap.add_argument('--scene_bbox_size', '--scene-bbox-size', dest='scene_bbox_size', type=float, default=2.6)
    ap.add_argument('--test_num', '--test-num', dest='test_num', type=int, default=10)
    ap.add_argument('--val_num', '--val-num', dest='val_num', type=int, default=0)
    ap.add_argument('--split_strategy', '--split-strategy', dest='split_strategy', default='tail')
    ap.add_argument('--cmaps', default='Fast')
    ap.add_argument('--band_count', '--band-count', dest='band_count', type=int, default=10)
    ap.add_argument('--shading', type=int, default=0)
    ap.add_argument('--opaque_unit', '--opaque-unit', dest='opaque_unit', type=float, default=1.5)
    ap.add_argument('--fxaa', type=int, default=1)
    ap.add_argument('--png_compress', '--png-compress', dest='png_compress', type=int, default=3)
    ap.add_argument('--tf_mode', '--tf-mode', dest='tf_mode', default='linear')
    ap.add_argument('--range', default='')
    ap.add_argument('--array_name', '--array-name', dest='array_name', default=None)
    ap.add_argument('--bg', default='0,0,0')
    ap.add_argument('--up', default='0,1,0')
    ap.add_argument('--opacity_only', action='store_true')
    ap.add_argument('--opacity_scale', '--opacity-scale', dest='opacity_scale', type=float, default=1.0)
    ap.add_argument('--hist_eq', action='store_true')
    ap.add_argument('--vector_magnitude', action='store_true')
    ap.add_argument('--gradient', action='store_true')
    ap.add_argument('--grad_opacity', '--grad-opacity', dest='grad_opacity', type=float, default=0.2)
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--adaptive_camera', '--adaptive-camera', dest='adaptive_camera', type=int, default=0)
    ap.add_argument('--anysplat_root', '--anysplat-root', dest='anysplat_root', default=None)
    ap.add_argument('--anysplat_only', '--anysplat-only', dest='anysplat_only', action='store_true')
    ap.add_argument('--camera_convention', '--camera-convention', dest='camera_convention',
                    choices=['opengl', 'opencv'], default='opengl')
    return ap

def main():
    args = build_parser().parse_args()
    from vol2splat.rendering.engine.exporter import MultiViewExporter
    MultiViewExporter(args).run()

if __name__ == '__main__':
    main()
