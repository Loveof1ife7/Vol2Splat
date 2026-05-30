# Simulation dataset 
(/root/autodl-tmp/conda_envs/data) root@autodl-container-2v5jqfbf4z-0e997d70:~/autodl-tmp/projects/data/datasets_for_volume_3dgs# ls
csafe_heptane_302x302x302_uint8           neghip_64x64x64_uint8
hcci_oh_560x560x560_float32               nucleon_41x41x41_uint8
hydrogen_atom_128x128x128_uint8           tacc_turbulence_256x256x256_float32_part_0000
miranda_1024x1024x1024_float32_part_0000  tacc_turbulence_256x256x256_float32_part_0001
miranda_1024x1024x1024_float32_part_0001  tacc_turbulence_256x256x256_float32_part_0002
miranda_1024x1024x1024_float32_part_0002  tacc_turbulence_256x256x256_float32_part_0003
miranda_1024x1024x1024_float32_part_0003  tacc_turbulence_256x256x256_float32_part_0004
miranda_1024x1024x1024_float32_part_0004  tacc_turbulence_256x256x256_float32_part_0005
miranda_1024x1024x1024_float32_part_0005  tacc_turbulence_256x256x256_float32_part_0006
miranda_1024x1024x1024_float32_part_0006  tacc_turbulence_256x256x256_float32_part_0007
miranda_1024x1024x1024_float32_part_0007

这是我的师弟制造的数据集合 我质疑它的质量，需要重新制作


# Dataset quality issues

## Core logic

We should define the failure case first, then choose metrics to describe it.

Target failure case:

- rendered image is visually `low-freq`
- rendered image is visually `foggy`
- pixel coverage may still be large
- therefore simple coverage / brightness checks are insufficient

So the logic is:

1. describe what bad render looks like
2. translate that visual property into measurable quantities
3. use those quantities for QC filtering

## Failure-case description

Current bad TFs are not necessarily:

- empty
- too dark
- too sparse

Instead, they often are:

- spatially broad
- alpha-supported
- but lacking sharp structure
- dominated by smooth, low-frequency content
- visually similar to mist / shell / cloudy mass

Representative example:

- case: `outputs/overfitting_usage/batch_0001/s0137`
- good: `TF02`, `TF03`, `TF06`
- bad: `TF04`, `TF05`

## Metric-design direction

Metrics should describe:

1. how much high-frequency content exists
2. how much local detail / sharpness exists
3. whether the visible foreground is detailed or just opaque and smooth

Current implemented metrics already move in this direction:

- `masked_fft_high_freq_ratio`
- `masked_laplacian_variance`
- `detail_over_opacity`

Interpretation:

- `masked_fft_high_freq_ratio`: measures whether the foreground has enough bandwidth
- `masked_laplacian_variance`: measures local sharpness / edge richness
- `detail_over_opacity`: penalizes cases that are opaque but structurally weak

## Current status

Completed:

1. extended render QC beyond coverage / brightness / contrast
2. added low-frequency / foggy failure-case metrics in `vol2splat/rendering/qc.py`
3. added regression test for "coverage is fine but render is foggy and low-freq"
4. updated config examples with new QC knobs

## Current result

For `outputs/overfitting_usage/batch_0001/s0137`:

- desired pass: `TF02`, `TF03`, `TF06`
- desired fail: `TF04`, `TF05`
- current trial result:
  - pass: `TF02`, `TF03`, `TF06`, `TF07`, `TF08`
  - fail: `TF01`, `TF04`, `TF05`

This means the failure-case description is partly captured correctly, but thresholds are not yet perfectly selective.

Threshold calibration update:

- old trial thresholds:
  - `min_masked_fft_high_freq_ratio: 0.012`
  - `min_detail_over_opacity: 0.08`
- tuned thresholds:
  - `min_masked_fft_high_freq_ratio: 0.0095`
  - `min_detail_over_opacity: 0.075`

Reason for tuning:

- `s0244/TF03` is visually a fairly good rendered image
- it was being rejected only because both metrics were slightly below the old thresholds
- with the tuned thresholds:
  - `s0244/TF03` returns to pass
  - `s0137/TF04` and `s0137/TF05` still fail

Foggy-rescue update:

- some renders are visually attractive and structurally meaningful
  even though their average `detail_over_opacity` is slightly low
- examples confirmed by inspection:
  - `s0703/TF02`
  - `s1073/TF02`

To preserve this kind of sample, QC now supports a conservative rescue path:

- only when the only failure reason is `foggy_low_detail`
- only when foreground alpha is not too heavy
- only when masked high-frequency content is still strong enough
- only when `detail_over_opacity` is close to the threshold rather than far below it

Current rescue config:

```yaml
rescue_foggy_foreground_alpha_max: 0.5
rescue_foggy_min_masked_fft_high_freq_ratio: 0.011
rescue_foggy_min_detail_over_opacity: 0.055
```

Observed effect on `batch_0001`:

- `s0703/TF02`: fail -> pass
- `s1073/TF02`: fail -> pass
- `s0137/TF04` and `s0137/TF05`: still fail

## Next tasks

1. refine the textual definition of `low-freq and foggy` so it is stable across cases
2. verify whether current metrics really align with that definition on more cases
3. continue calibrating thresholds so `TF04` / `TF05` fail while avoiding unnecessary rejection like `TF01`
4. verify whether the new rescue rule is still conservative enough on more batches
5. decide whether hard thresholds plus rescue are enough, or whether ranking is better
6. optionally add more interpretable summary metrics specifically for:
   - fogginess
   - edge richness
   - foreground bandwidth

## Latest validated rescue set on `overfitting_usage`

Current target rescue cases confirmed by manual inspection:

- `batch_0002/s1237/TF03`
- `batch_0002/s0045/TF02`
- `batch_0002/s0059/TF02`
- `batch_0001/s0244/TF02`
- `batch_0003/s0575/TF05`
- `batch_0003/s1103/TF06`

Current explicit non-rescue case:

- `batch_0003/s1103/TF07`
  - reason: too sparse / too dark
  - interpretation: structure spikes exist, but coverage is too small to keep

Current broader rescue rule now also preserves these additional cases:

- `batch_0001/s0703/TF02`
- `batch_0001/s1073/TF02`
- `batch_0002/s0686/TF02`
- `batch_0002/s1195/TF02`
- `batch_0003/s0575/TF03`
- `batch_0003/s0946/TF02`

Validation update:

- user confirmed that the additional rescued cases above are also good renders
- this supports that the current decision rule is capturing the intended concept correctly:
  - preserve soft-but-structured cases
  - reject sparse / thick-fog / low-bandwidth bad cases

Next review focus:

1. check whether the extra rescued cases above are all visually acceptable
2. confirm whether `peak-structure rescue` is now broad enough without leaking obvious bad fog cases
3. keep `s1103/TF07` as the negative control for "high-frequency spike but insufficient spatial coverage"

## Passed-TF export

Need:

- extract only `passed_tf_names` from `outputs/overfitting_usage`
- preserve original hierarchy:
  - `qc/batch_0001/<case_id>/<TFxx>`
  - `qc/batch_0002/<case_id>/<TFxx>`
  - `qc/batch_0003/<case_id>/<TFxx>`
- use soft links instead of copying data

Implemented:

- script: `scripts/link_passed_tf_from_qc.py`
- default input: `<dataset-root>/qc_lowfreq_summary.json`
- default output: `<dataset-root>/qc`
- rebuild option: `--clean-output-root`

Current generated result:

- output root: `outputs/overfitting_usage/qc`
- linked passed TF count: `131`
- summary file: `outputs/overfitting_usage/qc/passed_tf_links_summary.json`
