import numpy as np
import math
from typing import Tuple, List
import argparse

# 从 utils 导入几何和相机辅助函数
from .utils import (
    _unit, parse_vec3, intrinsics_from_vfov, 
    safe_viewup, c2w_from_lookat
)

class Scene:
    """管理场景几何、归一化和相机位姿。"""
    def __init__(self, bounds: Tuple[float], args: argparse.Namespace):
        print("[Scene] Initializing scene geometry and cameras...")
        self.args = args
        self.up_world = _unit(parse_vec3(args.up, (0,1,0)))
        
        # 1. 计算 BBox 和归一化参数
        if not bounds or not np.isfinite(bounds).all():
            raise RuntimeError("无法获取数据包围盒。")
            
        b = bounds
        self.bounds = b
        self.center = np.array([(b[0]+b[1])/2.0, (b[2]+b[3])/2.0, (b[4]+b[5])/2.0], float)
        self.ext = np.array([b[1]-b[0], b[3]-b[2], b[5]-b[4]], float)

        self.scene_bbox_size = float(getattr(args, "scene_bbox_size", 2.6))
        self.scale_factor = 1.0
        self.offset = np.zeros(3, dtype=float)
        
        print("-" * 50)
        print(f"[info] Render-world BBox Center: {self.center.tolist()}")
        print(f"[info] Original BBox Extents: {self.ext.tolist()}")
        print(f"[info] Target scene bbox size: {self.scene_bbox_size:.6f}")
        print("-" * 50)

        # 2. 计算内参
        fx, fy, cx, cy = intrinsics_from_vfov(float(args.vfov), int(args.w), int(args.h))
        vfov_rad = math.radians(float(args.vfov))
        hfov_rad = 2.0 * math.atan((float(args.w)/float(args.h)) * math.tan(vfov_rad/2.0))
        
        self.intrinsics = {
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "w": int(args.w), "h": int(args.h),
            "vfov": vfov_rad, "hfov": hfov_rad
        }

        # 3. 生成相机位姿
        self.poses = self._generate_fibonacci_poses(vfov_rad, hfov_rad)

    def _generate_fibonacci_poses(self, vfov: float, hfov: float) -> List[Tuple]:
        """计算相机距离并生成Fibonacci球面采样位姿。"""
        # 计算 fit 距离
        dx, dy, dz = map(float, self.ext.tolist())
        r_sphere = 0.5 * float(np.linalg.norm([dx, dy, dz])) # 外接球半径
        eps = 1e-6
        d_v = r_sphere / max(math.sin(vfov/2.0), eps)
        d_h = r_sphere / max(math.sin(hfov/2.0), eps)
        d_fit = max(d_v, d_h)
        d = float(self.args.radius_scale) * d_fit
        print(f"[info] fit distance d_fit={d_fit:.4f}, use d={d:.4f} (radius_scale={self.args.radius_scale})")

        # Fibonacci 采样
        N = int(self.args.num)
        phi = (1.0 + 5**0.5) / 2.0
        ga = 2.0 * math.pi * (1.0 - 1.0/phi)
        poses = []

        for i in range(N):
            u = (i + 0.5) / N
            z = 2.0*u - 1.0
            r_xy = max(0.0, 1.0 - z*z) ** 0.5
            theta = i * ga

            dirx = r_xy * math.cos(theta)
            diry = r_xy * math.sin(theta)
            dirz = z

            eye = [self.center[0] + d*dirx, self.center[1] + d*diry, self.center[2] + d*dirz]
            focal = self.center.tolist()
            
            forward = np.array(focal) - np.array(eye)
            vup = safe_viewup(forward, self.up_world)
            poses.append((eye, focal, vup.tolist()))
            
        print(f"[Scene] Generated {len(poses)} camera poses.")
        return poses

    def get_normalized_c2w(self, eye: List[float], focal: List[float], vup: List[float]) -> np.ndarray:
        """输入相机已经位于 render world，直接导出 render-world c2w。"""
        return c2w_from_lookat(eye, focal, vup)
    
    def generate_focused_poses(self, target_center: List[float], target_radius: float) -> List[Tuple]:
        """
        生成一组聚焦于特定中心和半径的相机位姿。
        注意：这不会改变全局归一化参数，只改变相机的位置。
        """
        # 1. 确定相机距离
        # 使用 args.radius_scale 来控制缩放余量
        # 如果 target_radius 太小（比如噪声），设置一个最小阈值，防止相机贴在脸上
        min_radius = np.max(self.ext) * 0.05 # 至少是全场景 5% 的大小
        safe_radius = max(target_radius, min_radius)
        
        # 根据 FOV 计算需要的距离
        vfov_rad = self.intrinsics['vfov']
        d_fit = safe_radius / math.sin(vfov_rad / 2.0)
        
        # 应用用户定义的缩放系数
        d = float(self.args.radius_scale) * d_fit
        
        print(f"[Scene] Generating focused poses. Target: {target_center}, Radius: {safe_radius:.2f}, Dist: {d:.2f}")

        # 2. Fibonacci 采样 (复用之前的逻辑，但中心和距离不同)
        N = int(self.args.num)
        phi = (1.0 + 5**0.5) / 2.0
        ga = 2.0 * math.pi * (1.0 - 1.0/phi)
        poses = []

        tc = np.array(target_center)

        for i in range(N):
            u = (i + 0.5) / N
            z = 2.0*u - 1.0
            r_xy = max(0.0, 1.0 - z*z) ** 0.5
            theta = i * ga

            dirx = r_xy * math.cos(theta)
            diry = r_xy * math.sin(theta)
            dirz = z
            
            # 方向向量
            direction = np.array([dirx, diry, dirz])
            
            # 相机位置 = 目标中心 + 距离 * 方向
            eye = tc + d * direction
            focal = tc # 聚焦于目标中心
            
            # 计算 Up 向量
            forward = focal - eye
            vup = safe_viewup(forward, self.up_world)
            
            poses.append((eye.tolist(), focal.tolist(), vup.tolist()))

        return poses
