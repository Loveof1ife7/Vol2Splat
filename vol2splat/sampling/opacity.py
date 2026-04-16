import numpy as np

from ..common.tf import bake_rgba_volume
from ..core.pipeline import Sampler
from ..core.types import Metadata, PointCloud, Volume
from ..io.read_vti import VTIReader
from ..registry import register_sampler
from .utils import apply_render_world_transform_if_present, resolve_render_world_transform, resolve_tf_json_path


class OpacitySampler(Sampler):
    def sample_canonical_vti(self, canonical_vti_path: str, cfg):
        vol = VTIReader().read(canonical_vti_path)
        return self.sample(vol, cfg)

    def sample(self, vol: Volume, cfg) -> PointCloud:
        n_points = int(cfg.get("n_points", 50000))
        tf_backend = cfg.get("tf_backend", "auto")
        tf_device = cfg.get("tf_device")
        premultiply_alpha = bool(cfg.get("premultiply_alpha", False))
        uniform_tf_filter = bool(cfg.get("uniform_tf_filter", False))
        uniform_stride_cfg = cfg.get("uniform_stride", 1)
        alpha_threshold = float(cfg.get("alpha_threshold", 0.0))
        use_jitter = bool(cfg.get("jitter", not uniform_tf_filter))

        if vol.data.ndim == 4 and vol.data.shape[0] == 4:
            rgba = np.asarray(vol.data, dtype=np.float32)
        else:
            tf_json = resolve_tf_json_path(cfg)
            rgba = bake_rgba_volume(
                vol.data,
                tf_json_path=tf_json,
                premultiply_alpha=premultiply_alpha,
                backend=tf_backend,
                device=tf_device,
                return_numpy=True,
            )

        alpha = np.asarray(rgba[3], dtype=np.float32)
        flat_alpha = np.clip(alpha.reshape(-1), 0.0, None)
        total = flat_alpha.shape[0]

        if uniform_tf_filter:
            z_dim, y_dim, x_dim = alpha.shape

            def _candidate_count_for_stride(s: int) -> int:
                return int(np.ceil(z_dim / s) * np.ceil(y_dim / s) * np.ceil(x_dim / s))

            if isinstance(uniform_stride_cfg, str):
                stride_raw = uniform_stride_cfg.strip().lower()
                auto_stride = stride_raw == "auto"
                uniform_stride = 1 if auto_stride else int(stride_raw)
            else:
                uniform_stride = int(uniform_stride_cfg)
                auto_stride = uniform_stride <= 0

            if auto_stride:
                keep_ratio = float(np.mean(flat_alpha > alpha_threshold))
                keep_ratio = max(keep_ratio, 1e-6)
                approx_candidates = float(n_points) / keep_ratio
                approx_stride = int(round((float(total) / max(approx_candidates, 1.0)) ** (1.0 / 3.0)))
                approx_stride = max(1, approx_stride)

                best_stride = approx_stride
                best_gap = float("inf")
                for s in range(max(1, approx_stride - 4), approx_stride + 5):
                    expected_kept = _candidate_count_for_stride(s) * keep_ratio
                    gap = abs(expected_kept - float(n_points))
                    if gap < best_gap:
                        best_gap = gap
                        best_stride = s
                uniform_stride = best_stride

            assert uniform_stride >= 1, "uniform_stride must be >= 1"
            z_idx = np.arange(0, z_dim, uniform_stride, dtype=np.int64)
            y_idx = np.arange(0, y_dim, uniform_stride, dtype=np.int64)
            x_idx = np.arange(0, x_dim, uniform_stride, dtype=np.int64)
            zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
            candidate = np.ravel_multi_index((zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)), dims=alpha.shape)
            kept = candidate[flat_alpha[candidate] > alpha_threshold]
            assert kept.size > 0, "No voxels remain after TF alpha filtering"
            target_n_points = min(n_points, int(kept.size))
            picked = np.random.choice(kept.size, size=target_n_points, replace=False)
            indices = kept[picked]
        else:
            probs = flat_alpha
            if probs.sum() <= 1e-12:
                probs = np.ones_like(probs, dtype=np.float32)
            probs = probs / probs.sum()
            nonzero_indices = np.flatnonzero(probs > 0)
            nonzero = int(nonzero_indices.shape[0])
            assert nonzero > 0, "No nonzero-probability voxels available for sampling"
            target_n_points = min(n_points, nonzero)
            weights = probs[nonzero_indices]
            weights = weights / weights.sum()
            picked = np.random.choice(nonzero, size=target_n_points, replace=False, p=weights)
            indices = nonzero_indices[picked]
        z_idx, y_idx, x_idx = np.unravel_index(indices, alpha.shape)
        z_idx_i = z_idx.astype(np.int64, copy=False)
        y_idx_i = y_idx.astype(np.int64, copy=False)
        x_idx_i = x_idx.astype(np.int64, copy=False)
        ijk = np.column_stack((x_idx, y_idx, z_idx)).astype(np.float32)
        if use_jitter:
            jitter = np.random.uniform(low=-0.5, high=0.5, size=ijk.shape).astype(np.float32)
            ijk = ijk + jitter
        xyz = vol.index_to_world(ijk).astype(np.float32)
        xyz, render_world_applied = apply_render_world_transform_if_present(xyz, cfg)

        scalar = np.asarray(vol.data if vol.data.ndim == 3 else rgba[3], dtype=np.float32)
        attrs = {
            "opacity": alpha.reshape(-1)[indices].reshape(-1, 1),
            "rgb": np.stack(
                [
                    rgba[0].reshape(-1)[indices],
                    rgba[1].reshape(-1)[indices],
                    rgba[2].reshape(-1)[indices],
                ],
                axis=1,
            ).astype(np.float32),
            "rgba": np.stack(
                [
                    rgba[0].reshape(-1)[indices],
                    rgba[1].reshape(-1)[indices],
                    rgba[2].reshape(-1)[indices],
                    rgba[3].reshape(-1)[indices],
                ],
                axis=1,
            ).astype(np.float32),
            "density": scalar.reshape(-1)[indices].reshape(-1, 1).astype(np.float32),
        }
        if bool(cfg.get("anisotropic_init", False)):
            from ..common.anisotropic_init import (
                anisotropic_scales_and_rots,
                apply_uniform_scale_to_log_scales,
                scalar_gradient_index_to_world,
            )

            field_name = str(cfg.get("aniso_field", "alpha")).strip().lower()
            if field_name == "alpha":
                field_zyx = alpha.astype(np.float32, copy=False)
            elif field_name == "scalar":
                if vol.data.ndim == 3:
                    field_zyx = np.asarray(vol.data, dtype=np.float32)
                else:
                    field_zyx = np.asarray(vol.data[0], dtype=np.float32)
            else:
                raise ValueError(f"aniso_field must be 'alpha' or 'scalar', got {field_name!r}")

            grad_zyx = scalar_gradient_index_to_world(vol, field_zyx)
            log_scales, rots, normals_n = anisotropic_scales_and_rots(
                vol,
                grad_zyx,
                z_idx_i,
                y_idx_i,
                x_idx_i,
                float(cfg.get("aniso_tangent_scale_mul", 1.5)),
                float(cfg.get("aniso_normal_scale_mul", 0.1)),
                str(cfg.get("aniso_voxel_size_mode", "mean_spacing")),
                float(cfg.get("aniso_grad_eps", 1e-6)),
                float(cfg.get("aniso_min_linear_scale", 1e-6)),
            )
            tr = resolve_render_world_transform(cfg)
            if tr is not None and "scale_factor" in tr:
                apply_uniform_scale_to_log_scales(log_scales, float(tr["scale_factor"]))
            attrs["scale_0"] = log_scales[:, 0:1]
            attrs["scale_1"] = log_scales[:, 1:2]
            attrs["scale_2"] = log_scales[:, 2:3]
            attrs["rot_0"] = rots[:, 0:1]
            attrs["rot_1"] = rots[:, 1:2]
            attrs["rot_2"] = rots[:, 2:3]
            attrs["rot_3"] = rots[:, 3:4]
            attrs["nx"] = normals_n[:, 0:1]
            attrs["ny"] = normals_n[:, 1:2]
            attrs["nz"] = normals_n[:, 2:3]

        metadata = Metadata(
            source_path=vol.metadata.source_path,
            original_dtype=vol.metadata.original_dtype,
            units=vol.metadata.units,
            extra=dict(vol.metadata.extra),
        )
        pc = PointCloud(xyz=xyz, attrs=attrs, metadata=metadata)
        pc.world_space = "render_world" if render_world_applied else "canonical_world"
        return pc


register_sampler("opacity", OpacitySampler)
