# Rendering Module

This module connects the main pipeline to the ParaView rendering engine.

## Layers

There are two layers inside `vol2splat/rendering`:

### 1. Pipeline wrapper layer

Files:

- `pv_engine.py`

This layer is responsible for:

- receiving `canonical_vti_path` from the main pipeline
- spawning `pvpython`
- forwarding renderer config as CLI flags
- discovering render outputs such as `TFxx/tf_config.json`
- returning render metadata to downstream stages

### 2. Engine layer

Files:

- `engine/run_exporter.py`
- `engine/exporter.py`
- `engine/volume_renderer.py`
- `engine/scene.py`
- `engine/tf_manager.py`

This layer is responsible for actual ParaView rendering.

## Input Contract

Rendering accepts canonical VTI only.

That means:

- the volume has already been canonicalized
- spacing is isotropic
- original scanner translation and orientation are no longer used downstream

Rendering should not need to know whether the source file was `nii.gz`, `raw`, or `vti`.

## World Spaces

### Canonical world

Produced by preprocess.

### Render world

Used during actual rendering and camera export.

The renderer computes a uniform transform from canonical bounds into a target bbox:

```text
render_world = canonical_world * scale_factor + offset
```

This transform is applied consistently to:

- the volume display in ParaView
- the scene bounds used for camera generation
- the exported camera matrices
- the point cloud export stage through pipeline metadata

So rendered RGB, exported poses, and exported point clouds all live in the same world.

## Why `pvpython` is launched as a subprocess

The pipeline wrapper does not import `paraview.simple` directly.

Instead it launches:

- `pvpython`
- `engine/run_exporter.py`

This keeps the main Python process lightweight and avoids coupling the whole project import path to ParaView runtime requirements.

## Output Contract

Typical output folder:

```text
outputs/scene_name/
├── TF01/
│   ├── train/
│   ├── test/
│   ├── val/
│   ├── tf_config.json
│   ├── transforms_train.json
│   └── transforms_test.json
├── TF02/
└── ...
```

Important outputs:

- `tf_config.json`
  Fused transfer function exported from ParaView.
  This is consumed later by opacity-aware sampling.

- `transforms_*.json`
  Camera poses in render world.

- PNG images
  Rendered training data for downstream models.

## Typical Flow

```text
canonical.vti
  -> pv_engine wrapper
  -> pvpython run_exporter.py
  -> engine/exporter.py
  -> TF folders + images + transforms + tf_config.json
```
