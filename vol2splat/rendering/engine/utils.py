# src/dataset_generator/utils.py
import json
import numpy as np
import re, math
from typing import List, Optional
# 确保 paraview 的导入在这里
from paraview.simple import *




def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def c2w_from_lookat(eye, center, up_world=(0,1,0)):
    eye, center, up_world = np.asarray(eye,float), np.asarray(center,float), _unit(up_world)
    forward = _unit(center - eye)
    right   = _unit(np.cross(forward, up_world))
    up      = _unit(np.cross(right, forward))
    c2w = np.eye(4, dtype=float)
    c2w[0:3,0] = right
    c2w[0:3,1] = up
    c2w[0:3,2] = -forward
    c2w[0:3,3] = eye
    return c2w

def intrinsics_from_vfov(vfov_deg, w, h):
    vfov = math.radians(vfov_deg)
    fy = (h/2.0) / math.tan(vfov/2.0)
    fx = fy * (w/float(h))
    cx, cy = (w-1)/2.0, (h-1)/2.0
    return fx, fy, cx, cy

def parse_vec3(s, default):
    if s is None: return np.array(default, float)
    parts = [p for p in s.replace(',', ' ').split() if p]
    if len(parts) != 3: return np.array(default, float)
    return np.array([float(x) for x in parts], float)

def parse_cmaps(s):
    if not s: return []
    # `cmaps` may already be a Python list from YAML, or a JSON-encoded list passed
    # through the CLI to preserve preset names that contain commas.
    if isinstance(s, list):
        return [str(item).strip() for item in s if str(item).strip()]
    if isinstance(s, str):
        text = s.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                items = json.loads(text)
                if isinstance(items, list):
                    return [str(item).strip() for item in items if str(item).strip()]
            except Exception:
                pass
    items = [t.strip() for t in re.split(r'[,\n]+', s) if t.strip()]
    seen, out = set(), []
    for it in items:
        if it not in seen:
            seen.add(it); out.append(it)
    return out

def sanitize_name(s):
    s = s.strip()
    s = re.sub(r'[\s/\\:;,\(\)]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "cmap"

def try_apply_preset(lut, name):
    ok = False
    try: ok = lut.ApplyPreset(name, True)
    except: pass
    if not ok:
        print("[warn] ApplyPreset 失败或未找到预设：", name)

def try_enable_nvidia_index(view, disp):
    # (函数体未改变)
    for attr in ["EnableNVIDIAIndeX", "EnableNVINDEX", "UseNVIDIAIndeX", "EnableIndeX"]:
        if hasattr(view, attr):
            try:
                setattr(view, attr, 1)
                print("[info] 视图属性已开启：", attr)
                break
            except:
                pass
    try:
        disp.SetRepresentationType('NVIDIA IndeX')
        print("[info] 已切换 Representation: NVIDIA IndeX")
        return True
    except:
        try:
            disp.Representation = 'NVIDIA IndeX'
            print("[info] 已切换 Representation: NVIDIA IndeX (fallback)")
            return True
        except:
            pass
    for rep in ['NVIDIA IndeX Volume', 'Volume (NVIDIA IndeX)']:
        try:
            disp.SetRepresentationType(rep)
            print("[info] 已切换 Representation:", rep)
            return True
        except:
            pass
    return False

def safe_viewup(forward, up_hint):
    # (函数体未改变)
    f = np.asarray(forward, float); f /= (np.linalg.norm(f)+1e-12)
    u = np.asarray(up_hint, float);  u /= (np.linalg.norm(u)+1e-12)
    if abs(np.dot(f, u)) > 0.98:
        alt = np.array([0,0,1.0]) if abs(f[2]) < 0.9 else np.array([0,1.0,0])
        u = alt - np.dot(alt, f) * f
    u = u - np.dot(u, f) * f
    u /= (np.linalg.norm(u)+1e-12)
    return u

def compute_hist_eq_band_edges(src, array_name, arr_min, arr_max, n_b):
    # (函数体未改变)
    from paraview import servermanager
    print("[info] 正在获取体数据标量值用于计算直方图...")
    data_obj = servermanager.Fetch(src)
    scalars = np.array(data_obj.GetPointData().GetArray(array_name))
    if scalars.size == 0:
        raise RuntimeError("体数据中没有有效体素。")
    print("[info] 正在计算直方图和CDF...")
    counts, bin_edges_hist = np.histogram(scalars, bins=2048, range=(arr_min, arr_max))
    cdf = np.cumsum(counts)
    total_voxels = cdf[-1]
    if total_voxels <= 0:
        raise RuntimeError("体数据体素总数为0。")
    print(f"[info] 计算 {n_b} 等体素分段...")
    quantiles = np.linspace(0, total_voxels, n_b + 1)
    split_indices = np.searchsorted(cdf, quantiles[1:-1], side='right')
    band_edges = [float(arr_min)] + [float(bin_edges_hist[i]) for i in split_indices] + [float(arr_max)]
    print("[info] 基于直方图计算出的新分段边界:")
    print("    " + ", ".join([f"{x:.3g}" for x in band_edges]))
    return band_edges
