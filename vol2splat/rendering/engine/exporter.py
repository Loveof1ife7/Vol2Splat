import os, json
import numpy as np
from typing import Tuple, List
import argparse

# 导入我们包中的其他类
from .volume_renderer import VolumeRenderer
from .scene import Scene
from .tf_manager import TFManager
#from .vtk_depth_renderer import render_depth_with_raymarching


class MultiViewExporter:
    """协调器：管理 Renderer, Scene, TFManager 并执行导出工作流。"""
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.renderer = VolumeRenderer(args.vti, args)
        self.scene = Scene(self.renderer.render_bounds, args)
        self.tf_manager = TFManager(args)
        # 初始化 anysplat 相关状态
        self.anysplat_scene_idx = 0
        self.anysplat_scene_names = []

    def run(self):
        """执行整个导出过程。"""
        print("\n[Exporter] Starting export run...")
        
        # 1. 计算直方图（如果需要）
        self.tf_manager.compute_histogram_bands(
            self.renderer.src, 
            self.renderer.array_name, 
            self.renderer.arr_min, 
            self.renderer.arr_max
        )
        # self.tf_manager.analyze_band_statistics(
        #     self.renderer.src,
        #     self.renderer.array_name,
        #     self.renderer.arr_min,
        #     self.renderer.arr_max
        # )
        # 2. 尝试启用 IndeX
        self.renderer.try_enable_index()

        # 3. 主循环
        print(f"\n[Exporter] Starting main render loop for {len(self.tf_manager.cmaps)} TF(s)...")
        for idx, cmap, tf_name in self.tf_manager:
            
            out_dir = os.path.join(self.args.out, tf_name)
            os.makedirs(out_dir, exist_ok=True)
            
            print(f"\n--- Processing TF {idx}/{len(self.tf_manager.cmaps)}: {tf_name} (CMap: {cmap}) ---")
            pts = self.tf_manager.make_tf_points(idx=idx, n_b=self.args.band_count, 
            arr_min=self.renderer.arr_min, arr_max=self.renderer.arr_max, 
                                 mode=self.args.tf_mode)
            # 3a. 应用 TF
            self.renderer.apply_tf(
                cmap=cmap, 
                idx=idx, 
                n_bands=int(self.args.band_count), 
               pts = pts,
                opacity_only=self.args.opacity_only
            )
            
            # 3b. 导出 TF JSON（若只关心 anysplat，可跳过）
            if not (getattr(self.args, "anysplat_root", None) and getattr(self.args, "anysplat_only", False)):
                self.renderer.export_fused_tf_json(out_dir)

            # 3c. 渲染和拆分数据集
            frames_train, frames_test, frames_val = self._render_and_split_dataset(tf_name, out_dir, pts, band_idx=idx) # <--- **修改这里**

            # 3d. 导出 Transforms JSON
            if getattr(self.args, "anysplat_root", None):
                # 如果指定了 anysplat_root，直接导出为 anysplat 格式
                self._export_anysplat_format(tf_name, frames_train, frames_test, frames_val)
            else:
                # 否则导出 Vol2GS 格式
                self._export_transforms_json(out_dir, frames_train, frames_test, frames_val)
            
        # 如果指定了 anysplat_root，在最后写入索引文件
        if getattr(self.args, "anysplat_root", None) and self.anysplat_scene_names:
            self._write_anysplat_index()
            
        print("\n[Exporter] All tasks completed.")

    def _render_and_split_dataset(self, tf_name: str, out_dir: str, pts: List[float], band_idx: int) -> Tuple[List, List, List]: # <--- **修改签名**
        
        """循环渲染所有位姿，应用归一化，保存图片并返回帧数据。"""
        frames_train, frames_test, frames_val = [], [], []
        
        # 检查是否直接保存为 anysplat 格式
        anysplat_root = getattr(self.args, "anysplat_root", None)
        if anysplat_root:
            # 直接保存到 anysplat 格式
            scene_name = f"scene{self.anysplat_scene_idx:04d}"
            self.anysplat_scene_idx += 1
            self.anysplat_scene_names.append(scene_name)
            scene_dir = os.path.join(anysplat_root, scene_name)
            images_dir = os.path.join(scene_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            # 仍然创建 Vol2GS 格式的目录（如果不需要 anysplat_only，可以用于深度图等）
            if not getattr(self.args, "anysplat_only", False):
                out_img_dir_train = os.path.join(out_dir, "train")
                out_img_dir_test = os.path.join(out_dir, "test")
                out_img_dir_val = os.path.join(out_dir, "val")
                out_img_dir_depth = os.path.join(out_dir, "depth")
                os.makedirs(out_img_dir_train, exist_ok=True)
                os.makedirs(out_img_dir_test, exist_ok=True)
                os.makedirs(out_img_dir_val, exist_ok=True)
            else:
                out_img_dir_train = None
                out_img_dir_test = None
                out_img_dir_val = None
                out_img_dir_depth = None
        else:
            # 保存为 Vol2GS 格式
            out_img_dir_train = os.path.join(out_dir, "train")
            out_img_dir_test = os.path.join(out_dir, "test")
            out_img_dir_val = os.path.join(out_dir, "val") 
            out_img_dir_depth = os.path.join(out_dir, "depth") 
            os.makedirs(out_img_dir_train, exist_ok=True)
            os.makedirs(out_img_dir_test, exist_ok=True)
            os.makedirs(out_img_dir_val, exist_ok=True)
            scene_dir = None
            images_dir = None

        # 计算数据集拆分点
        current_poses = self.scene.poses # 默认：全局位姿
        pose_type = "Global"
        
        # 检查是否开启自适应，并且该 Band 有空间信息
        if getattr(self.args, 'adaptive_camera', 0) > 0:
            if hasattr(self.tf_manager, 'band_spatial_info') and band_idx in self.tf_manager.band_spatial_info:
                info = self.tf_manager.band_spatial_info[band_idx]
                # 生成聚焦位姿
                current_poses = self.scene.generate_focused_poses(info['center'], info['radius'])
                pose_type = f"Adaptive (R={info['radius']:.2f})"
            else:
                # 如果没有空间信息（比如该band被keep但算不出包围盒），回退到全局
                print(f"[Exporter] No spatial info for TF{band_idx}, using Global poses.")
        

        print(f"[Exporter] Camera Strategy: {pose_type}")
        # =========================================================

        # 计算数据集拆分点
        Nposes = len(current_poses) # 使用 current_poses 的长度
        test_num = int(self.args.test_num) if int(self.args.test_num) >= 0 else 0
        val_num = int(self.args.val_num) if int(self.args.val_num) >= 0 else 0
        if test_num + val_num > Nposes:
            if test_num > Nposes:
                test_num, val_num = Nposes, 0
            else:
                val_num = max(0, Nposes - test_num)
        
        def _even_indices(total, k):
            if k <= 0 or total <= 0:
                return []
            if k >= total:
                return list(range(total))
            lin = np.linspace(0, total - 1, k, endpoint=False)
            return sorted({int(round(x)) for x in lin})

        split_strategy = getattr(self.args, "split_strategy", "tail")
        if split_strategy == "interleave":
            all_idx = list(range(Nposes))
            test_idx = _even_indices(Nposes, test_num)
            remaining = [i for i in all_idx if i not in test_idx]
            val_rel = _even_indices(len(remaining), val_num)
            val_idx = [remaining[i] for i in val_rel]
            train_idx = [i for i in all_idx if i not in test_idx and i not in val_idx]
        else:
            test_start = Nposes - test_num
            val_start = Nposes - test_num - val_num
            if val_start < 0: val_start = 0
            test_idx = list(range(test_start, Nposes))
            val_idx = list(range(val_start, test_start))
            train_idx = list(range(0, val_start))

        print(f"[info] TF={tf_name}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} (split={split_strategy})")
        bounds_world = self.scene.bounds

        for i, (eye, focal, vup) in enumerate(current_poses):
            # 1. 渲染 (保持不变)
            self.renderer.render_at_pose(eye, focal, vup, self.args.vfov)
            near_plane, far_plane = self.renderer.get_camera_clipping_range()

            # 2. 获取归一化的 C2W 矩阵 (保持不变)
            c2w = self.scene.get_normalized_c2w(eye, focal, vup)
            
            # 3. 分配和保存 (略微简化 RGB 路径逻辑)
            if anysplat_root:
                # 直接保存为 anysplat 格式
                frame_idx = len(frames_train) + len(frames_val) + len(frames_test)
                fname = f"frame_{frame_idx:04d}.png"
                fpath_rgb = os.path.join(images_dir, fname)
                self.renderer.save_screenshot(fpath_rgb, int(self.args.w), int(self.args.h), int(self.args.png_compress))
                
                # 确定属于哪个数据集（用于后续索引）
                if i in test_idx:
                    split_tag = "test"
                    frames_list = frames_test
                elif i in val_idx:
                    split_tag = "val"
                    frames_list = frames_val
                else:
                    split_tag = "train"
                    frames_list = frames_train
                
                # 如果不需要 anysplat_only，也保存到 Vol2GS 格式目录
                if not getattr(self.args, "anysplat_only", False):
                    vol2gs_fname = f"r_{i:04d}.png"
                    if split_tag == "test":
                        vol2gs_dir = out_img_dir_test
                    elif split_tag == "val":
                        vol2gs_dir = out_img_dir_val
                    else:
                        vol2gs_dir = out_img_dir_train
                    vol2gs_path = os.path.join(vol2gs_dir, vol2gs_fname)
                    self.renderer.save_screenshot(vol2gs_path, int(self.args.w), int(self.args.h), int(self.args.png_compress))
                    print(f"Saved {split_tag} frame (both formats): {fpath_rgb} and {vol2gs_path}")
                else:
                    print(f"Saved {split_tag} frame (anysplat): {fpath_rgb}")
            else:
                # 保存为 Vol2GS 格式
                fname = f"r_{i:04d}.png"
                if i in test_idx:
                    out_img_dir = os.path.join(out_dir, "test")
                    split_tag = "test"
                    frames_list = frames_test
                elif i in val_idx:
                    out_img_dir = os.path.join(out_dir, "val")
                    split_tag = "val"
                    frames_list = frames_val
                else:
                    out_img_dir = os.path.join(out_dir, "train")
                    split_tag = "train"
                    frames_list = frames_train
                
                # 保存 RGB 图像
                fpath_rgb = os.path.join(out_img_dir, fname)
                self.renderer.save_screenshot(fpath_rgb, int(self.args.w), int(self.args.h), int(self.args.png_compress))
                print(f"Saved {split_tag} frame: {fpath_rgb}")

            # 确定深度图输出的基础路径（必须在 "depth" 目录下）
            if out_img_dir_depth:
                file_base = f"r_{i:04d}"
                # 深度图路径: out_dir/depth/r_0000_depth.npy
                file_base_path = os.path.join(out_img_dir_depth, file_base)
            else:
                file_base_path = None 

            # 4. **深度图生成：强制使用 Contour 渲染（支持所有格式）**
            # 我们已弃用 extract_depth_map，统一使用 render_depth_with_contour
            # 现在直接在 ParaView RenderView 中操作，确保与 RGB 使用完全相同的视角
            # npy_path = None
            # if file_base_path:
            #     try:
            #         print(f"[Exporter] 开始渲染第 {i} 帧的深度图...")
            #         npy_path = render_depth_with_raymarching(
            #             renderer_obj=self.renderer,
            #             output_base_path=file_base_path,
            #             opacity_points=pts,
            #             tf_band_idx=band_idx,
            #             save_debug_image=(i == 0),
            #             save_color_depth=True,
            #         )
            #         if npy_path:
            #             print(f"[Exporter] 第 {i} 帧深度图渲染成功: {npy_path}")
            #         else:
            #             print(f"[Exporter] 警告: 第 {i} 帧深度图渲染返回 None")
            #     except Exception as e:
            #         import traceback
            #         print(f"[Exporter] 错误: 渲染第 {i} 帧深度图时发生异常: {type(e).__name__}: {str(e)}")
            #         traceback.print_exc()
            #         if i == 0:
            #             print(f"[Exporter] 深度图生成失败（第一帧），将继续处理后续帧...")

            # if npy_path is None and i == 0:
            #     print(f"[Exporter] 警告: 第一帧深度图生成失败，仅导出 RGB。")


            
            # 5. 更新帧数据
            if anysplat_root:
                # anysplat 格式：file_path 相对于 scene 目录
                frame_data = {
                    "file_path": os.path.join("images", fname).replace("\\", "/"),
                    "transform_matrix": c2w.tolist(),
                }
            else:
                # Vol2GS 格式
                relative_file_base = f"./{split_tag}/{file_base}"
                frame_data = {
                    "file_path": relative_file_base,
                    "transform_matrix": c2w.tolist(),
                }
                # if npy_path:
                #     # 深度文件相对于 out_dir
                #     frame_data["depth_path"] = os.path.join("./depth", f"r_{i:04d}_depth.npy").replace("\\", "/")
                #     print(f"Saved depth to: {frame_data['depth_path']}")
                
            frames_list.append(frame_data)
                
        return frames_train, frames_test, frames_val

    def _export_transforms_json(self, out_dir: str, frames_train: List, frames_test: List, frames_val: List):
        """为 train/test/val 数据集生成 transforms.json 文件。

        如果当前处于 anysplat_only 模式，则不在 Vol2GS 目录下额外写 transforms_*，
        只保留 anysplat 所需的 transforms.json（由上层逻辑负责）。
        """
        if getattr(self.args, "anysplat_root", None) and getattr(self.args, "anysplat_only", False):
            return
        meta_base = {
            "camera_angle_x": self.scene.intrinsics["hfov"],
            "w": self.scene.intrinsics["w"], "h": self.scene.intrinsics["h"],
            "fl_x": self.scene.intrinsics["fx"], "fl_y": self.scene.intrinsics["fy"], 
            "cx": self.scene.intrinsics["cx"], "cy": self.scene.intrinsics["cy"],
            "camera_model": "OPENCV",
            "render_bounds": list(self.scene.bounds),
            "render_world_transform": self.renderer.render_world_transform,
        }
        # (此函数逻辑与之前相同)
        if frames_train:
            meta_train = meta_base.copy(); meta_train["frames"] = frames_train
            tf_path_train = os.path.join(out_dir, "transforms_train.json")
            with open(tf_path_train, "w", encoding="utf-8") as f:
                json.dump(meta_train, f, indent=2, ensure_ascii=False)
            print("Wrote", tf_path_train)
        if frames_test:
            meta_test = meta_base.copy(); meta_test["frames"] = frames_test
            tf_path_test = os.path.join(out_dir, "transforms_test.json")
            with open(tf_path_test, "w", encoding="utf-8") as f:
                json.dump(meta_test, f, indent=2, ensure_ascii=False)
            print("Wrote", tf_path_test)
        if frames_val:
            meta_val = meta_base.copy(); meta_val["frames"] = frames_val
            tf_path_val = os.path.join(out_dir, "transforms_val.json")
            with open(tf_path_val, "w", encoding="utf-8") as f:
                json.dump(meta_val, f, indent=2, ensure_ascii=False)
            print("Wrote", tf_path_val)

    def _export_anysplat_format(self, tf_name: str, frames_train: List, frames_test: List, frames_val: List):
        """导出 anysplat 格式的 transforms.json 文件。
        
        将所有帧（train/test/val）合并到一个 transforms.json 中。
        """
        anysplat_root = getattr(self.args, "anysplat_root", None)
        if not anysplat_root:
            return
        
        # 获取当前场景名称（最后一个）
        if not self.anysplat_scene_names:
            return
        scene_name = self.anysplat_scene_names[-1]
        scene_dir = os.path.join(anysplat_root, scene_name)
        os.makedirs(scene_dir, exist_ok=True)
        
        # 合并所有帧
        all_frames = frames_train + frames_val + frames_test
        
        if not all_frames:
            print(f"[warn] scene {scene_name} 没有有效帧，跳过。")
            self.anysplat_scene_names.pop()
            self.anysplat_scene_idx -= 1
            return
        
        # 构建 anysplat 期望的 transforms.json
        transforms_any = {
            "fl_x": self.scene.intrinsics["fx"],
            "fl_y": self.scene.intrinsics["fy"],
            "cx": self.scene.intrinsics["cx"],
            "cy": self.scene.intrinsics["cy"],
            "w": self.scene.intrinsics["w"],
            "h": self.scene.intrinsics["h"],
            "frames": all_frames,
        }
        
        out_json = os.path.join(scene_dir, "transforms.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(transforms_any, f, indent=2, ensure_ascii=False)
        
        print(f"[info] 写入 anysplat scene: {scene_name}, 帧数={len(all_frames)} (train={len(frames_train)}, val={len(frames_val)}, test={len(frames_test)})")

    def _write_anysplat_index(self):
        """写入 anysplat 的 train_index.json 和 test_index.json 文件。"""
        anysplat_root = getattr(self.args, "anysplat_root", None)
        if not anysplat_root or not self.anysplat_scene_names:
            return
        
        os.makedirs(anysplat_root, exist_ok=True)
        
        train_index_path = os.path.join(anysplat_root, "train_index.json")
        test_index_path = os.path.join(anysplat_root, "test_index.json")
        
        with open(train_index_path, "w", encoding="utf-8") as f:
            json.dump(self.anysplat_scene_names, f, indent=2, ensure_ascii=False)
        
        # 先简单复制一份为 test，后续可手动调整
        with open(test_index_path, "w", encoding="utf-8") as f:
            json.dump(self.anysplat_scene_names, f, indent=2, ensure_ascii=False)
        
        print(f"[info] 写入 train_index.json / test_index.json，场景数量 = {len(self.anysplat_scene_names)}")
