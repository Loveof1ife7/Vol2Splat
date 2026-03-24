# Sampling Module

This module converts canonical volumes into point clouds.

## Main Idea

Sampling happens after canonicalization, and often after rendering has already generated transfer functions.

The sampling stage therefore supports two broad modes:

### 1. Scalar-volume sampling

Examples:

- `density`
- `gradient`
- `uniform`
- `poisson`
- `hybrid`

These operate directly on scalar volume values or scalar-derived importance signals.

### 2. Transfer-function-aware sampling

Example:

- `opacity`

This mode reads `tf_config.json`, bakes the scalar volume into RGBA volume, and samples according to opacity or RGBA-aware importance.

## Input Contract

Sampling consumes either:

- in-memory `Volume`
- or `canonical_vti_path` through `sample_canonical_vti(...)`

The expected volume contract is:

- canonical voxel world
- isotropic spacing
- scalar data as `(Z, Y, X)`
- RGBA data as `(4, Z, Y, X)`

## Transfer Function Flow

When opacity-aware sampling is used, the flow is:

```text
canonical scalar volume
  + tf_config.json
  -> bake RGBA volume
  -> sample points
  -> export render-world point cloud
```

Shared TF logic lives in:

- `vol2splat/common/tf.py`

Sampling-specific TF discovery helpers live in:

- `vol2splat/sampling/utils.py`

## Multi-TF Behavior

If rendering produced multiple transfer functions:

```text
outputs/scene/TF01/tf_config.json
outputs/scene/TF02/tf_config.json
...
```

and the sampler is `opacity`, the main pipeline can automatically sample all discovered TFs if no explicit selector is given.

This leads to outputs such as:

```text
outputs/scene/TF01/points.ply
outputs/scene/TF02/points.ply
...
```

## World Contract

Sampling itself may produce points in canonical world.

Before writing final output, export can apply the same render-world transform used by the renderer:

```text
exported_xyz = canonical_xyz * scale_factor + offset
```

This keeps:

- rendered images
- camera poses
- exported point clouds

in one consistent target bbox world.

## Samplers

- `density`
  Sample according to scalar magnitude distribution.

- `gradient`
  Sample high-gradient voxels to preserve edges and structures.

- `uniform`
  Random baseline sampler.

- `poisson`
  Approximate blue-noise style spatial sampling.

- `hybrid`
  Combine structure-aware and random sampling.

- `opacity`
  Use TF-baked alpha as the main sampling distribution.
