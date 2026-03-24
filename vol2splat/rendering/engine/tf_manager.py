import re
import numpy as np
import argparse
from typing import List, Optional, Set
# 从 utils 导入 TF 相关的辅助函数
from .utils import parse_cmaps, compute_hist_eq_band_edges

class TFManager:
    """管理传递函数（TF）的生成和迭代。"""
    def __init__(self, args: argparse.Namespace):
        print("[TFManager] Initializing TF sets...")
        self.args = args
        self.band_edges_data: Optional[List[float]] = None
        self.skipped_bands: Set[int] = set()
        self.skipped_bands = set()
        
        cmaps = parse_cmaps(args.cmaps)
        if not cmaps:
            cmaps = ["Viridis (matplotlib)", "Inferno (matplotlib)", "Plasma (matplotlib)", "Magma (matplotlib)", "Turbo"]
            print("[info] 未指定 --cmaps，使用默认集合：", ", ".join(cmaps))

        self.tf_names: List[str] = []
        
        if int(args.band_count) > 0:
            n_b = int(args.band_count)
            self.tf_names = [f"TF{i:02d}" for i in range(1, n_b + 1)]
            if len(cmaps) == 0:
                self.cmaps = ["Inferno (matplotlib)"] * n_b
            elif len(cmaps) < n_b:
                self.cmaps = (cmaps * ((n_b // len(cmaps)) + 1))[:n_b]
            else:
                self.cmaps = cmaps[:n_b]
            print(f"[info] band_count enabled: generating {n_b} TF bands")
        else:
            self.cmaps = cmaps
            self.tf_names = [f"TF{i:02d}" for i in range(1, len(self.cmaps) + 1)]
            
        print("[info] cmaps to process:", ", ".join(self.cmaps))

    def compute_histogram_bands(self, src, array_name: str, arr_min: float, arr_max: float):
        """如果 --hist_eq 启用，则计算分段边界。"""
        if int(self.args.band_count) > 0 and self.args.hist_eq:
            print("[TFManager] --hist_eq enabled, calculating band edges...")
            try:
                self.band_edges_data = compute_hist_eq_band_edges(src, array_name, arr_min, arr_max, int(self.args.band_count))
            except Exception as e:
                print(f"[warn] Failed to compute histogram-equalized band edges: {e}. Falling back to linear bands.")
                self.band_edges_data = None # 确保回退

    def __iter__(self):
        """迭代 (idx, cmap_name, tf_name)。"""
        for idx, (cmap, tf_name) in enumerate(zip(self.cmaps, self.tf_names), start=1):
            if idx in self.skipped_bands:
                continue # 跳过 yield
            yield idx, cmap, tf_name
    
    def make_tf_points(self,idx: int,n_b: int,arr_min: float,arr_max: float,
                    mode: str = "unit") -> List[float]:
        """
        生成透明度传输函数点（Points）。
        mode: "unit" (矩形), "linear" (三角形), "gaussian" (高斯)
        """
        band_edges=self.band_edges_data
        # 1. 确定当前 band 的边界 (low, high)
        # (此逻辑不变)
        if band_edges:
            # 如果我们已经成功计算了基于直方图的边界，直接使用它们
            low = band_edges[idx-1]
            high = band_edges[idx]
        else:
            # 否则，使用原始的线性分割逻辑作为回退
            span = float(arr_max - arr_min) if arr_max > arr_min else 1.0
            band_w = span / float(n_b)
            low = float(arr_min) + float(idx-1) * band_w
            high = float(arr_min) + float(idx) * band_w

        # 2. 构造点元组 (x, alpha) 列表
        pts_tuples = []
        
        if mode == "unit": 
            # 矩形带 (修正原始逻辑以确保边界清晰)
            eps = (high - low) * 0.001 # 使用一个更小的 epsilon
            
            pts_tuples.append((float(arr_min), 0.0))
            if low > float(arr_min):
                pts_tuples.append((low, 0.0))
            
            pts_tuples.append((low + eps, 0.2)) # 阶跃上升
            pts_tuples.append((high - eps, 0.2)) # 保持
            
            if high < float(arr_max):
                pts_tuples.append((high, 0.0)) # 阶跃下降
            pts_tuples.append((float(arr_max), 0.0))
            
            print(f"[info] TF band {idx}/{n_b} (unit): [{low:.6g}, {high:.6g}]")

        elif mode == "linear": 
            # 折线样条 (三角形)
            mid = (low + high) / 2.0
            width = (high - low)
            delta = max(0.05 * width, 1e-9)  # 5% shoulder

            pts_tuples.append((float(arr_min), 0.0))
            if low > float(arr_min):
                pts_tuples.append((low, 0.0)) # band 开始点

            pts_tuples.append((low + delta, 0.08))    
            pts_tuples.append((mid, 0.2)) # band 中心点 (峰值)
            pts_tuples.append((high - delta, 0.08))

            if high < float(arr_max):
                pts_tuples.append((high, 0.0)) # band 结束点
            pts_tuples.append((float(arr_max), 0.0))
            
            print(f"[info] TF band {idx}/{n_b} (linear): [{low:.6g}, {high:.6g}], peak at {mid:.6g}")

        elif mode == "gaussian":
            mean = (low + high) / 2.0
            width = high - low
            # 4-sigma 逻辑：在 low 和 high 处，alpha 约 0.135
            std = max(width / 4.0, 1e-9) 
            
            # 在 band 内采样 64 个点以创建平滑曲线
            N_SAMPLES = 64 
            
            # 使用一个极小值来创建清晰的边界
            eps = (arr_max - arr_min) * 0.0001 
            
            pts_tuples = []
            
            # 1. 最小值到 band 开始前
            pts_tuples.append((float(arr_min), 0.0))
            if low > float(arr_min):
                pts_tuples.append((low - eps, 0.0)) # 在 band 开始前保持为 0
            
            # 2. 生成平滑的高斯曲线
            # 使用 np.linspace 来确保包含 low 和 high 点
            sample_points = np.linspace(low, high, N_SAMPLES, dtype=float) 
            for x in sample_points:
                z = (x - mean) / std
                alpha = math.exp(-0.5 * z * z)
                pts_tuples.append((float(x), alpha))
                
            # 3. band 结束后
            if high < float(arr_max):
                # 在 high 点的值由 linspace 保证
                pts_tuples.append((high + eps, 0.0)) # 在 band 结束后立刻降为 0
                
            pts_tuples.append((float(arr_max), 0.0))
            
            print(f"[info] TF band {idx}/{n_b} (gaussian): mean={mean:.6g}, std={std:.6g} over [{low:.6g}, {high:.6g}]")

        else:
            print(f"[warn] 未知的 TF mode: {mode}。返回空 TF。")
            return []

        # 3. --- 去除重复点并转换为ParaView格式 ---
        # (此逻辑不变)
        final_pts = []
        last_x = -float('inf')
        for x, a in pts_tuples:
            # 确保 x 值是单调递增的
            x_val = max(x, last_x + 1e-9) 
            if abs(x_val - last_x) > 1e-9:
                final_pts.append((x_val,a))
                last_x = x_val
            else: 
                final_pts[-1] = (x_val,a) # 如果 x 值重复，用后面的点覆盖

        # 转换为 ParaView.simple.Points 格式 [x, y, midpoint, sharpness]
        pts = []
        for x, a in final_pts:
            pts.extend([float(x), float(a), 0.5, 0.0])
            
        return pts
    
    def make_tf_points_grouped(self, idx_list: List[int], n_b: int,
                            arr_min: float, arr_max: float,
                            mode: str = "linear") -> List[float]:
        all_pts = []
        for idx in idx_list:
            pts = self.make_tf_points(idx, n_b, arr_min, arr_max, mode)
            all_pts.extend(pts)

        pts_array = np.array(all_pts).reshape(-1, 4)
        pts_array = pts_array[pts_array[:, 0].argsort()] # sort by x
        return pts_array.flatten().tolist()
    
    
    def get_current_opacity_points(self, idx: int, arr_min: float, arr_max: float) -> List[float]:
        """
        重新计算并返回当前 TF 的 Opacity Points 列表。
        此方法用于纯 VTK 深度渲染器的 Opacity TF 设置。
        """
        # 直接调用 make_tf_points 来生成并返回 [x, y, midpoint, sharpness] 格式的列表
        # 这里需要知道当前正在处理的 cmap 是哪一个，以便获取 mode
        
        # 假设 make_tf_points 只需要 idx, n_b, arr_min, arr_max, mode
        n_b = int(self.args.band_count)
        mode = self.args.tf_mode
        
        # 由于 make_tf_points 内部已经包含了所有的计算和逻辑，我们直接调用它
        return self.make_tf_points(
            idx=idx, 
            n_b=n_b, 
            arr_min=arr_min, 
            arr_max=arr_max,
            mode=mode
        )
    
    def analyze_band_statistics(self, src, array_name: str, arr_min: float, arr_max: float):
            """
            统计每个 Band 的体素数量、占比，并计算其空间包围盒（中心和半径）。
            """
            print("\n[TFManager] Analyzing voxel statistics and spatial distribution...")
            
            # 1. 获取数据对象及几何信息 (Dimensions, Spacing, Origin)
            # ----------------------------------------------------------------
            try:
                from paraview import servermanager
                # 尝试作为 ParaView Proxy Fetch
                # 注意：我们需要 Fetch 后的对象才能调用 GetDimensions 等 VTK 方法
                if hasattr(src, 'GetDataInformation'): 
                    data_obj = servermanager.Fetch(src)
                else:
                    data_obj = src # 假设已经是 vtkImageData
                    
                from vtk.util import numpy_support
                
                # 获取几何信息 (关键步骤)
                dims = data_obj.GetDimensions()   # (nx, ny, nz)
                spacing = data_obj.GetSpacing()   # (dx, dy, dz)
                origin = data_obj.GetOrigin()     # (ox, oy, oz)

                # 获取标量数组
                vtk_array = data_obj.GetPointData().GetArray(array_name)
                if vtk_array is None:
                    vtk_array = data_obj.GetPointData().GetScalars()
                    
                if vtk_array is None:
                    print("[warn] Could not fetch data array. Skipping analysis.")
                    return

                scalars = numpy_support.vtk_to_numpy(vtk_array)
                
            except Exception as e:
                print(f"[warn] Error analyzing statistics: {e}. Skipping analysis.")
                return

            total_voxels = scalars.size
            if total_voxels == 0: return

            n_b = int(self.args.band_count)
            if n_b <= 0: return

            # 阈值设置
            min_pct = getattr(self.args, 'min_voxel_percent', 0.0001) 
            max_pct = getattr(self.args, 'max_voxel_percent', 1.00) 

            # === 初始化空间信息存储 ===
            # 结构: { band_idx: {'center': [x,y,z], 'radius': r} }
            self.band_spatial_info = {} 
            self.skipped_bands.clear()

            print(f"{'Band':<4} | {'Range':<20} | {'Voxels':<10} | {'%':<8} | {'Status':<15} | {'Center & Radius'}")
            print("-" * 95)

            nx, ny, nz = dims
            
            for idx in range(1, n_b + 1):
                # 获取边界
                if self.band_edges_data:
                    low = self.band_edges_data[idx-1]
                    high = self.band_edges_data[idx]
                else:
                    span = (arr_max - arr_min)
                    band_w = span / float(n_b)
                    low = arr_min + (idx-1) * band_w
                    high = arr_min + idx * band_w

                # 生成掩码
                if idx == n_b: mask = (scalars >= low) & (scalars <= high)
                else:          mask = (scalars >= low) & (scalars < high)

                count = np.sum(mask)
                percent = count / total_voxels
                
                status = "KEEP"
                spatial_str = ""

                # 过滤逻辑
                if percent < min_pct:
                    status = "SKIP (Sparse)"
                    self.skipped_bands.add(idx)
                elif percent > max_pct:
                    status = "SKIP (Dense)"
                    self.skipped_bands.add(idx)
                else:
                    # === 核心：计算包围盒与中心 ===
                    # 获取该 Band 所有体素的平坦索引
                    valid_indices = np.flatnonzero(mask)
                    
                    if valid_indices.size > 0:
                        # 将平坦索引转换为 (x, y, z) 网格索引
                        # VTK/NumPy 的 flatten 顺序通常是: x 变化最快, 然后 y, 然后 z
                        # i_x = idx % nx
                        # i_y = (idx // nx) % ny
                        # i_z = idx // (nx * ny)
                        
                        ix = valid_indices % nx
                        iy = (valid_indices // nx) % ny
                        iz = valid_indices // (nx * ny)
                        
                        # 找到 Index Space 的包围盒
                        min_ix, max_ix = np.min(ix), np.max(ix)
                        min_iy, max_iy = np.min(iy), np.max(iy)
                        min_iz, max_iz = np.min(iz), np.max(iz)
                        
                        # 转换为 World Space 的中心
                        # Center = Origin + Spacing * (Min_Index + Max_Index) / 2
                        cx = origin[0] + spacing[0] * (min_ix + max_ix) / 2.0
                        cy = origin[1] + spacing[1] * (min_iy + max_iy) / 2.0
                        cz = origin[2] + spacing[2] * (min_iz + max_iz) / 2.0
                        
                        # 计算 World Space 的尺寸 (Extent)
                        extent_x = (max_ix - min_ix) * spacing[0]
                        extent_y = (max_iy - min_iy) * spacing[1]
                        extent_z = (max_iz - min_iz) * spacing[2]
                        
                        # 计算外接球半径 (Bounding Radius)
                        # 0.5 * 对角线长度
                        radius = 0.5 * np.linalg.norm([extent_x, extent_y, extent_z])
                        
                        # 存起来
                        self.band_spatial_info[idx] = {
                            'center': [cx, cy, cz],
                            'radius': radius
                        }
                        spatial_str = f"C=[{cx:.1f},{cy:.1f},{cz:.1f}] R={radius:.1f}"

                print(f"TF{idx:02d} | [{low:6.2f}, {high:6.2f}] | {count:<10} | {percent*100:6.3f}% | {status:<15} | {spatial_str}")

            print("-" * 95)
            print(f"[TFManager] Analysis done. Skipping {len(self.skipped_bands)} bands.\n")