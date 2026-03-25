import numpy as np
from paraview.simple import *
from typing import List, Tuple
import argparse
import os
import json 
from ...common.spatial import compute_uniform_target_bbox_transform
# 从我们的 utils 模块导入辅助函数
from .utils import (
    parse_vec3, try_apply_preset,
    try_enable_nvidia_index
)
from vtk import vtkGPUVolumeRayCastMapper, vtkVolume, vtkImageData
from vtk.util import numpy_support

# from PIL import Image # 保持这个用于可视化
class VolumeRenderer:
    """管理 ParaView 状态、流水线和渲染操作。"""
    def __init__(self, vti_path: str, args: argparse.Namespace):
        print("[Renderer] Initializing ParaView pipeline...")
        self.args = args
        # 确保每次新实例都在干净的 ParaView 会话中，避免残留上一个数据集的几何
        try:
            from paraview.simple import ResetSession
            ResetSession()
        except Exception:
            pass
        # 可以是 .vti、.xmf/.xdmf（内部引用 .h5），或者其它 ParaView 支持的体数据格式
        self.src = OpenDataFile(vti_path)
        if self.src is None: 
            raise RuntimeError("无法打开体数据文件：" + vti_path)

        # 1. 设置视图 (RV)
        self.rv = GetActiveViewOrCreate('RenderView')
        self.rv.ViewSize = [int(args.w), int(args.h)]
        bg = parse_vec3(args.bg, (0,0,0))
        try: self.rv.UseColorPaletteForBackground = 0
        except: pass
        self.rv.Background = [float(bg[0]), float(bg[1]), float(bg[2])]
        self.rv.CameraParallelProjection = 0
        self.rv.CameraViewAngle = float(args.vfov)
        if args.fxaa:
            try: self.rv.UseFXAA = 1
            except: pass
            
        try: self.rv.OrientationAxesVisibility = 0
        except: pass
        try: self.rv.EnableRayTracing = 0
        except: pass

        print("[Renderer] Reading data file (UpdatePipeline)...")
        UpdatePipeline() 
        print("[Renderer] Data loaded.")
        
        # 2. 处理 CellData 到 PointData，并根据需要对向量场取模长
        self.src, self.array_name, assoc = self._setup_data_array(self.src, self.rv)
        
        # 3. 获取 DisplayProperties (Disp)
        self.disp = GetDisplayProperties(self.src, view=self.rv)
        self.disp.Representation = 'Volume'
        ColorBy(self.disp, (assoc, self.array_name))

        # 4. 获取 LUT 和 PWF 并设置标量范围
        self.lut = GetColorTransferFunction(self.array_name)
        self.pwf = GetOpacityTransferFunction(self.array_name)
        self.arr_min, self.arr_max = self._setup_scalar_range(self.src, args.range)
        
        # 5. 隐藏标量条
        self.disp.SetScalarBarVisibility(self.lut, False)

        # 6. 获取体渲染范围
        raw_bounds = self.src.GetDataInformation().GetBounds()
        self.render_world_transform = compute_uniform_target_bbox_transform(
            raw_bounds,
            float(getattr(self.args, "scene_bbox_size", 2.6)),
        )
        self.render_bounds = tuple(self.render_world_transform["render_bounds"])
        scale = float(self.render_world_transform["scale_factor"])
        offset = self.render_world_transform["offset"]
        try:
            self.disp.Scale = [scale, scale, scale]
            self.disp.Position = [float(offset[0]), float(offset[1]), float(offset[2])]
        except Exception as exc:
            print(f"[warn] Failed to apply render-world transform on display: {exc}")

            
        # 7. 设置体渲染属性
        self.disp.Shade = int(args.shading)
        target_unit = float(self.args.opaque_unit)

        # 如果用户设为 -1 或 0，则自动计算
        if target_unit <= 0:
            from paraview import servermanager
            vtk_data = servermanager.Fetch(self.src)
            
            auto_unit = None

            # 优先：若是规则网格 (vtkImageData)，使用 spacing
            if hasattr(vtk_data, "GetSpacing"):

                spacing = vtk_data.GetSpacing()
                valid_spacing = [s for s in spacing if s and s > 0]
                if valid_spacing:
                    auto_unit = sum(valid_spacing) / len(valid_spacing)
                    print(f"[Renderer] Auto-set ScalarOpacityUnitDistance to: {auto_unit:.6f} (from ImageData spacing)")


            # 其次：对非 vtkImageData，基于包围盒和点数做一个估计
            if auto_unit is None:
                try:
                    bounds = vtk_data.GetBounds() if hasattr(vtk_data, "GetBounds") else None
                    npts = vtk_data.GetNumberOfPoints() if hasattr(vtk_data, "GetNumberOfPoints") else 0
                    if bounds and npts > 0:
                        # 估计一个等体素立方体的边长：用包围盒对角线长度 / N^(1/3)
                        diag = ((bounds[1]-bounds[0])**2 +
                                (bounds[3]-bounds[2])**2 +
                                (bounds[5]-bounds[4])**2) ** 0.5
                        cell_size = diag / max(float(npts) ** (1.0/3.0), 1e-6)
                        auto_unit = float(cell_size)
                        print(f"[Renderer] Auto-set ScalarOpacityUnitDistance to: {auto_unit:.6f} (from bounds & point count)")
                except Exception as e:
                    print(f"[warn] Failed to estimate ScalarOpacityUnitDistance from bounds/points: {e}")

            # 最后兜底：使用 1.0 并给出警告
            if auto_unit is None or not np.isfinite(auto_unit) or auto_unit <= 0:
                auto_unit = 1.0
                print("[warn] Fallback ScalarOpacityUnitDistance to 1.0 (could not infer from data).")

            target_unit = auto_unit
                
        self.disp.ScalarOpacityUnitDistance = float(target_unit)


        
        UpdatePipeline()
        Render()
        print(f"[Renderer] Pipeline ready. Array: '{self.array_name}', Range: [{self.arr_min}, {self.arr_max}]")
        print(f"[Renderer] Render-world bounds: {self.render_bounds}")

    def _setup_data_array(self, src, rv):
        """查找用于渲染/统计的数组。

        步骤：
        1. 如果指定了 --array_name，则优先按名称在 PointData / CellData 中查找；
        2. 否则使用 PointData 中的第一个数组；如果没有，则从 CellData 拷贝到 PointData；
        3. 如果选中的数组是多分量向量，并且开启了 --vector_magnitude，则通过 Calculator 计算其模长，生成新的标量数组。
        """
        di = src.GetDataInformation()
        pdi = di.GetPointDataInformation()
        cdi = di.GetCellDataInformation()

        array_name = None
        assoc = 'POINTS'
        used_c2p = False

        target_name = getattr(self.args, "array_name", None)

        def _find_array(info, name: str):
            if not info or info.GetNumberOfArrays() == 0:
                return None
            for i in range(info.GetNumberOfArrays()):
                ai = info.GetArrayInformation(i)
                if name is None or ai.GetName() == name:
                    return ai
            return None

        # 1. 优先在 PointData 中按名称查找
        ai = _find_array(pdi, target_name)

        # 2. 如果 PointData 中没找到，尝试在 CellData 中查找并做 CellData -> PointData
        if ai is None:
            ai_cell = _find_array(cdi, target_name)
            if ai_cell is not None:
                array_name = ai_cell.GetName()
                c2p = CellDatatoPointData(Input=src)
                Hide(src, rv)
                Show(c2p, rv)
                src = c2p
                UpdatePipeline()
                di = src.GetDataInformation()
                pdi = di.GetPointDataInformation()
                ai = _find_array(pdi, array_name)
                assoc, used_c2p = 'POINTS', True
        
        # 3. 仍然没找到时，回退为 PointData 第一个数组；若没有，则从 CellData 拷贝第一个
        if ai is None:
            if pdi and pdi.GetNumberOfArrays() > 0:
                ai = pdi.GetArrayInformation(0)
                array_name = ai.GetName()
            elif cdi and cdi.GetNumberOfArrays() > 0:
                ai_cell = cdi.GetArrayInformation(0)
                array_name = ai_cell.GetName()
                c2p = CellDatatoPointData(Input=src)
                Hide(src, rv)
                Show(c2p, rv)
                src = c2p
                UpdatePipeline()
                di = src.GetDataInformation()
                pdi = di.GetPointDataInformation()
                ai = _find_array(pdi, array_name)
                assoc, used_c2p = 'POINTS', True
            else:
                raise RuntimeError("数据中不存在任何 PointData / CellData 数组，无法渲染。")
        else:
            array_name = ai.GetName()

        # 4. 如果数组是多分量，并且启用了 vector_magnitude，则通过 Calculator 取模长
        num_comps = ai.GetNumberOfComponents() if ai is not None else 1
        use_vec_mag = getattr(self.args, "vector_magnitude", False)
        if use_vec_mag and num_comps > 1:
            print(f"[info] 选中的数组 '{array_name}' 是 {num_comps} 分量向量，将计算其模长用于渲染。")
            try:
                calc = Calculator(Input=src)
                mag_name = f"{array_name}_mag"
                calc.ResultArrayName = mag_name
                calc.Function = f"mag({array_name})"
                Hide(src, rv)
                Show(calc, rv)
                src = calc
                UpdatePipeline()

                di = src.GetDataInformation()
                pdi = di.GetPointDataInformation()
                ai = _find_array(pdi, mag_name)
                if ai is None:
                    raise RuntimeError("Calculator 生成的模长数组未能在 PointData 中找到。")
                array_name = mag_name
                assoc = 'POINTS'
            except Exception as e:
                print(f"[warn] 计算向量模长失败，回退为直接使用原始数组：{e}")

        print("[info] use array:", assoc, array_name, 
              "(CellData->PointData)" if used_c2p else "",
              "(vector magnitude)" if use_vec_mag and num_comps > 1 else "")
        return src, array_name, assoc

    def _setup_scalar_range(self, src, range_override_str: str) -> Tuple[float, float]:
        """获取标量范围，处理覆盖。"""
        override_range = None
        if range_override_str:
            ss = [t for t in range_override_str.replace(',', ' ').split() if t]
            if len(ss) == 2: override_range = (float(ss[0]), float(ss[1]))

        if override_range is not None:
            arr_min, arr_max = override_range
        else:
            # --- 修正：使用正确的信息 API ---
            di = src.GetDataInformation()
            pdi = di.GetPointDataInformation()
            ai = None
            if pdi and pdi.GetNumberOfArrays() > 0:
                for i in range(pdi.GetNumberOfArrays()):
                    # 比较数组名称
                    if pdi.GetArrayInformation(i).GetName() == self.array_name:
                        ai = pdi.GetArrayInformation(i)
                        break
            
            if ai is not None:
                try: arr_min, arr_max = ai.GetRange()
                except: arr_min, arr_max = ai.GetComponentRange(0)
            else:
                # 如果找不到数组（虽然不应该发生），给个默认值
                arr_min, arr_max = 0.0, 255.0
            # --- 修正结束 ---

        if not np.isfinite([arr_min, arr_max]).all() or arr_min == arr_max:
            arr_min, arr_max = 0.0, 255.0

        self.lut.RescaleTransferFunction(float(arr_min), float(arr_max))
        return float(arr_min), float(arr_max)

    def apply_tf(self, cmap: str, idx: int, n_bands: int, pts, opacity_only: bool):
        """应用指定的 Colormap 和 Opacity 设置。"""
        # 1. 设置 Colormap
        if opacity_only:
            if idx == 1: # 仅第一次设置
                try_apply_preset(self.lut, cmap)
                self.lut.RescaleTransferFunction(self.arr_min, self.arr_max)
        else:
            try_apply_preset(self.lut, cmap)
            self.lut.RescaleTransferFunction(self.arr_min, self.arr_max)

        # 2. 设置 Opacity
        if n_bands > 0:

            self.pwf.Points = pts
            try: 
                self.pwf.AllowDuplicateScalars = 1
            except: pass
        else:
            # TODO: 实现 args.opacity_preset, args.opacity_scale 等逻辑
            pass
        
        UpdatePipeline()
        Render()

    def apply_tf_from_paraview_json(self, path):
        with open(path) as f:
            tf = json.load(f)[0]

        try:
            self.lut.RemoveAllPoints()
        except AttributeError:
            # Try alternative method names
            if hasattr(self.lut, 'RemoveAllControlPoints'):
                self.lut.RemoveAllControlPoints()
            elif hasattr(self.lut, 'RemoveControlPoints'):
                self.lut.RemoveControlPoints()
        # self.lut.RemoveControlPoints()
        self.lut.RGBPoints = tf["RGBPoints"]
        self.lut.ColorSpace = tf.get("ColorSpace", "RGB")
        try:
            self.pwf.RemoveAllPoints()
        except AttributeError:
            # Try alternative method names
            if hasattr(self.pwf, 'RemoveAllControlPoints'):
                self.pwf.RemoveAllControlPoints()
            elif hasattr(self.pwf, 'RemoveControlPoints'):
                self.pwf.RemoveControlPoints()
        #self.pwf.RemoveAllPoints()
        self.pwf.Points = tf["Points"]
        self.pwf.AllowDuplicateScalars = 1

        self.lut.RescaleTransferFunction(self.arr_min, self.arr_max)
        UpdatePipeline(); Render()

    def export_fused_tf_json(self, out_dir: str):
        """融合 LUT 和 PWF 并导出为高精度 JSON。"""
        color_points_raw = list(self.lut.RGBPoints)
        opacity_points_raw = list(self.pwf.Points)

        if not color_points_raw or not opacity_points_raw:
            print("[warn] LUT或PWF为空，无法导出fused TF JSON。")
            return

        color_x_np = np.array([color_points_raw[i] for i in range(0, len(color_points_raw), 4)])
        color_r_np = np.array([color_points_raw[i+1] for i in range(0, len(color_points_raw), 4)])
        color_g_np = np.array([color_points_raw[i+2] for i in range(0, len(color_points_raw), 4)])
        color_b_np = np.array([color_points_raw[i+3] for i in range(0, len(color_points_raw), 4)])
        
        opacity_x_np = np.array([opacity_points_raw[i] for i in range(0, len(opacity_points_raw), 4)])
        opacity_a_np = np.array([opacity_points_raw[i+1] for i in range(0, len(opacity_points_raw), 4)])
        
        all_x_coords = np.union1d(color_x_np, opacity_x_np)
        
        tf_data = []
        for x in all_x_coords:
            r = np.interp(x, color_x_np, color_r_np)
            g = np.interp(x, color_x_np, color_g_np)
            b = np.interp(x, color_x_np, color_b_np)
            a = np.interp(x, opacity_x_np, opacity_a_np)
            tf_data.append([float(x), float(r), float(g), float(b), float(a)])

        tf_config = {
            "data_range": [self.arr_min, self.arr_max],
            "control_points": tf_data 
        }
        
        config_path = os.path.join(out_dir, "tf_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(tf_config, f, indent=2)
        print(f"[info] Exported *fused high-fidelity* TF to: {config_path}")

    def render_at_pose(self, eye: List[float], focal: List[float], vup: List[float], vfov: float):
        """设置相机并渲染。"""
        self.rv.CameraPosition = eye
        self.rv.CameraFocalPoint = focal
        self.rv.CameraViewUp = vup
        self.rv.CameraViewAngle = float(vfov)
        try: self.rv.ResetCameraClippingRange()
        except: pass
        Render()

    def save_screenshot(self, fpath: str, w: int, h: int, compression: int):
        """保存当前视图。"""
        SaveScreenshot(fpath, self.rv, ImageResolution=[w, h], 
                       CompressionLevel=compression, TransparentBackground=1)

    def try_enable_index(self):
        if not self.args.index: return
        print("[Renderer] Attempting to enable NVIDIA IndeX...")

        index_on = try_enable_nvidia_index(self.rv, self.disp)
        if not index_on:
            print("[warn] Failed to enable IndeX, fallback.")
            return

        self.disp.Shade = 1
        self.disp.Ambient = 0.15
        self.disp.Diffuse = 0.85
        self.disp.Specular = 0.5
        self.disp.SpecularPower = 50.0
        # --- IndeX gradient-based opacity tuning ---
        if hasattr(self.disp, "GradientOpacityLevel"):
            self.disp.GradientOpacityLevel = 50
        else:
            print("[warn] GradientOpacityLevel not supported")

        if hasattr(self.disp, "GradientOpacityScale"):
            self.disp.GradientOpacityScale = 20
        else:
            print("[warn] GradientOpacityScale not supported")

        if hasattr(self.disp,"ScalarOpacityUnitDistance"):
            self.disp.ScalarOpacityUnitDistance=self.args.opaque_unit
        else:
            print("[warn] ScalarOpacityUnitDistance not supported")
        #print(dir(self.disp))


        # === Manifold switch ===
        manifold = (self.args.tf_mode in ["manifold", "gaussian", "linear"])

        if hasattr(self.disp, "UseGradientForTransfer2D"):
            self.disp.UseGradientForTransfer2D = 1 if manifold else 0
            print("[Renderer] 2D Gradient TF:", self.disp.UseGradientForTransfer2D)

        # === Opacity integration thickness (critical) ===
        self.disp.ScalarOpacityUnitDistance = float(self.args.opaque_unit)

        ResetCamera(self.rv)
        Render()

        print("[Renderer] IndeX enabled. Manifold =", manifold)

    def extract_depth_map(self, fpath_base: str, save_vis: bool = True) -> Tuple[str, str]:
        try:
            w, h = self.rv.ViewSize
            
            # --- 1. 获取底层的 VTK Mapper (最终尝试) ---
            
            # 1a. 获取 ParaView DisplayProperties 对象 (self.disp) 的 'Mapper' 属性
            # 这是使用代理的通用方法，绕过直接属性访问失败的问题
            mapper_property = self.disp.GetProperty("Mapper") 
            
            if not mapper_property:
                 raise RuntimeError("无法获取 DisplayProperties 的 'Mapper' 属性对象。")

            # 1b. 从属性对象中获取底层的 Proxy (Mapper 代理)
            mapper_proxy = mapper_property.GetProxy() 
            
            if not mapper_proxy:
                 raise RuntimeError("无法从 'Mapper' 属性中获取代理对象。")

            # 1c. 从 ParaView 代理中获取底层的 VTK 对象 (这是 vtkGPUVolumeRayCastMapper 实例)
            mapper = mapper_proxy.GetVTKObject() 
            
            if not mapper:
                 raise RuntimeError("底层 VTK Mapper 对象为 None。")
                 
            # 确保导入了正确的 VTK 类进行类型检查
            from vtk import vtkGPUVolumeRayCastMapper 
            
            if not isinstance(mapper, vtkGPUVolumeRayCastMapper):
                 raise TypeError(f"Mapper type not supported: {type(mapper)}. 必须是 vtkGPUVolumeRayCastMapper。")

            # --- 2. 启用深度输出模式 ---
            mapper.SetRenderToImage(True)
            mapper.SetDepthImageScalarTypeToFloat() # 确保深度以 Float 精度输出
            
            # 强制重新渲染以捕获深度纹理
            self.rv.GetRenderWindow().Render()
            # --- 3. 提取深度图像 ---
            # GetDepthImage() 返回一个 vtkImageData 对象，其中包含 ray-casted 的深度
            depth_image_data = mapper.GetDepthImage() 
            
            if depth_image_data is None or depth_image_data.GetPointData().GetScalars() is None:
                 raise RuntimeError("vtkGPUVolumeRayCastMapper returned no depth image data. Check GPU/driver compatibility.")
            
            # 提取标量数组
            z_data_vtk = depth_image_data.GetPointData().GetScalars()
            z_buffer_array = numpy_support.vtk_to_numpy(z_data_vtk)
            
            # 调整数组形状和方向
            z_buffer_array = z_buffer_array.reshape(h, w)
            z_buffer_array = np.flipud(z_buffer_array) 

            # --- 4. 转换 Z_view (此时 Z_buffer_array 已经是视空间深度 Z_view) ---
            # vtkGPUVolumeRayCastMapper::GetDepthImage() 返回的通常是 ray-casted 视空间深度 (非归一化)
            Z_view = z_buffer_array.copy()
            
            # 由于 vtkGPUVolumeRayCastMapper::GetDepthImage() 通常输出非归一化的视空间深度，
            # 我们可以跳过反归一化步骤。但需要处理背景/空区域 (通常用远平面或一个大值表示)
            
            cam = self.rv.GetRenderer().GetActiveCamera()
            near_plane = cam.GetClippingRange()[0]
            far_plane = cam.GetClippingRange()[1]
            
            # 检查背景值（通常是 0.0 或一个极大值，这里假设它应该在 near/far 之间，否则是背景）
            Z_view[Z_view < near_plane] = far_plane # 任何小于 near 的深度值视作背景 (设为 far)
            
            # --- 5. 关闭深度输出模式 (重要!) ---
            mapper.SetRenderToImage(False) 
            
            # --- 6. 保存和可视化 (使用 Z_view) ---
            npy_path = fpath_base + "_depth.npy"
            npy_dir = os.path.dirname(npy_path)
            if npy_dir and not os.path.exists(npy_dir):
                os.makedirs(npy_dir, exist_ok=True)
            np.save(npy_path, Z_view.astype(np.float32))

            vis_png_path = None
            if save_vis:
                # 归一化用于可视化
                Z_vis_normalized = (Z_view - near_plane) / (far_plane - near_plane)
                Z_vis_normalized = np.clip(Z_vis_normalized, 0, 1)
                
                # 转换到 8-bit 可视化
                Z_vis_8bit = (1.0 - Z_vis_normalized) * 255.0 
                
                from PIL import Image
                vis_img = Image.fromarray(Z_vis_8bit.astype(np.uint8))
                
                vis_png_path = fpath_base + "_depth_vis.png"
                vis_img.save(vis_png_path)
            
            return npy_path, vis_png_path

        except Exception as e:
            # 如果深度提取失败，尝试关闭 SetRenderToImage 以免影响后续渲染
            try:
                mapper = self.disp.GetMapper().GetVTKObject()
                if hasattr(mapper, 'SetRenderToImage'):
                     mapper.SetRenderToImage(False)
            except:
                 pass
            print(f"[ERROR] 提取深度图失败：{e}")
            return None, None

    def get_camera_clipping_range(self) -> Tuple[float, float]:
            """获取当前的相机裁剪范围 (Near, Far)"""
            # 必须在调用 self.rv.ResetCameraClippingRange() 之后调用
            cam = self.rv.GetRenderer().GetActiveCamera()
            return cam.GetClippingRange()
