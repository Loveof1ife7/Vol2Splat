# 03_WORKLOG.md
> This file tracks current actionable tasks only.
> Move finished work, experiment logs, and validated conclusions to `02_WORKLOG.md`.

## Priority Definition
- `P0` — blocking issue
- `P1` — mainline task
- `P2` — improvement
- `P3` — optional exploration

## Status Definition
- `todo`      # 尚未开始
- `doing`     # 正在进行
- `blocked`   # 被依赖项、bug、资源等阻塞
- `validate`  # 实现已完成，正在验证
- `done`      # 已完成并确认
- `abandon`   # 放弃

---

## 2026-05-14 Render QC Upgrade

### Goal

Improve automatic dataset filtering for rendered TF variants.

The intended reasoning is:

1. define the visual failure case
2. describe that failure case with metrics
3. use those metrics for QC

This is not just "add more heuristics".

### Failure-case definition

The target bad case is:

- not necessarily dark
- not necessarily sparse
- may still have substantial pixel coverage
- but visually appears `low-freq and foggy`

More concretely, these renders tend to be:

- smooth
- broad
- lacking edge-rich internal structure
- dominated by low-frequency content
- visually like mist / shell / cloudy opacity instead of meaningful anatomy or structure

Observed examples:

- `outputs/overfitting_usage/batch_0001/s0137/TF04`
- `outputs/overfitting_usage/batch_0001/s0137/TF05`

### Why old QC was insufficient

Old render QC mainly checked:

- `mean_intensity`
- `nonzero_ratio`
- `intensity_std`

These can reject empty or too-dark TFs, but they do not directly describe:

- bandwidth
- edge richness
- whether foreground opacity contains real structure

So they miss the actual failure mode we care about.

### Metric mapping

The visual description `low-freq and foggy` was translated into measurable signals.

Current added metrics:

- `gradient_mean`
- `laplacian_variance`
- `fft_high_freq_ratio`
- `masked_laplacian_variance`
- `masked_fft_high_freq_ratio`
- `detail_over_opacity`

Interpretation of the most relevant ones:

- `masked_fft_high_freq_ratio`
  captures whether the visible foreground has enough high-frequency content

- `masked_laplacian_variance`
  captures whether the visible foreground has enough local sharpness / edge structure

- `detail_over_opacity`
  captures whether the render is detailed relative to how opaque it is
  and helps detect "opaque but structureless" cases

### Code changes

Updated:

- `vol2splat/rendering/qc.py`
- `tests/test_batch_qc_tiles.py`
- `configs/lqb/canonical_3dgs.yaml`
- `configs/lqb/batch_canonical_3dgs.yaml`

### Validation

Regression tests:

- `python -m unittest tests.test_batch_qc_tiles.TestBatchQCTiles`
- result: pass

Environment note:

- `conda activate ffgs` works for QC execution
- `pytest` is not installed there, so validation used `unittest`

Case check on `s0137` with:

```yaml
min_masked_fft_high_freq_ratio: 0.012
min_detail_over_opacity: 0.08
```

Observed result:

- pass: `TF02`, `TF03`, `TF06`, `TF07`, `TF08`
- fail: `TF01`, `TF04`, `TF05`

### Current conclusion

What is validated:

1. the correct abstraction is "failure case = low-freq and foggy"
2. this failure case should be described by metrics, not guessed from coverage alone
3. the new QC metrics successfully catch `TF04` and `TF05`

What is still open:

1. current thresholds are still somewhat aggressive
2. `TF01` is also rejected, so selectivity needs improvement
3. the verbal definition of `low-freq and foggy` still needs broader validation across more cases

### Threshold tuning update

Follow-up inspection showed that `s0244/TF03` is a false negative under the earlier thresholds.

Why it was misclassified:

- visually it is not in the same category as `s0244/TF04`
- but across sampled views its mean metrics landed just below the previous cutoff:
  - `masked_fft_high_freq_ratio = 0.009546`
  - `detail_over_opacity = 0.075516`
- previous thresholds were:
  - `min_masked_fft_high_freq_ratio: 0.012`
  - `min_detail_over_opacity: 0.08`

This means the rejection was caused by threshold aggressiveness, not by a clear mismatch with the intended visual concept.

Tuned thresholds:

```yaml
min_masked_fft_high_freq_ratio: 0.0095
min_detail_over_opacity: 0.075
```

Observed effect on `batch_0001`:

- `s0137` remains:
  - pass: `TF02`, `TF03`, `TF06`, `TF07`, `TF08`
  - fail: `TF01`, `TF04`, `TF05`
- `s0244` becomes:
  - pass: `TF03`, `TF05`, `TF06`, `TF07`, `TF08`
  - fail: `TF01`, `TF02`, `TF04`

Important observation:

- for `batch_0001`, the tuning changes only one decision:
  - `s0244/TF03`: `fail -> pass`

This is a good sign because it fixes the identified false negative without altering the intended behavior on `s0137`.

### Foggy rescue update

Further manual inspection identified two more visually strong renders that were still being rejected:

- `s0703/TF02`
- `s1073/TF02`

These two differ from truly foggy failures in an important way:

- they are soft on average
- but they still retain meaningful foreground structure
- and their foreground alpha is not as heavy as the more opaque fog-like failures

This suggested that the plain average-threshold rule was still too strict.

Instead of lowering global thresholds again, QC was extended with a conservative rescue rule.

Rescue is applied only if:

1. the only failure reason is `foggy_low_detail`
2. `foreground_alpha_mean` is below a maximum
3. `masked_fft_high_freq_ratio` is above a minimum
4. `detail_over_opacity` is still near the threshold rather than deeply below it

Configured values:

```yaml
rescue_foggy_foreground_alpha_max: 0.5
rescue_foggy_min_masked_fft_high_freq_ratio: 0.011
rescue_foggy_min_detail_over_opacity: 0.055
```

Observed effect on `batch_0001`:

- `s0703/TF02`: `fail -> pass`
- `s1073/TF02`: `fail -> pass`
- `s0137/TF04`: still `fail`
- `s0137/TF05`: still `fail`

This is preferable to another global threshold drop because it keeps the rule tied to the visual interpretation:

- soft but still structured foreground can be preserved
- thick opaque fog is still filtered out

### Next step

Use more cases to test whether the current metrics genuinely align with the visual concept:

- good render = structured, edge-rich, not overly smooth
- bad render = broad, smooth, low-bandwidth, foggy

Then decide whether to keep:

1. global thresholds
2. dataset-specific thresholds
3. rank-based pruning instead of pure thresholding

### Passed-TF symlink export

Need:

- extract only QC-passed TFs from `outputs/overfitting_usage`
- keep the original batch / case / TF hierarchy
- avoid data copy by using symlinks

Implemented script:

- `scripts/link_passed_tf_from_qc.py`

Default behavior:

1. read dataset summary from `<dataset-root>/qc_lowfreq_summary.json`
2. create output tree at `<dataset-root>/qc`
3. mirror each passed TF as:
   - `qc/<batch_id>/<case_id>/<TFxx> -> original TF directory`

Added regression test:

- `tests/test_link_passed_tf_from_qc.py`

Validated run on the real dataset:

```bash
python scripts/link_passed_tf_from_qc.py \
  --dataset-root outputs/overfitting_usage \
  --clean-output-root
```

Observed output:

- created `outputs/overfitting_usage/qc`
- total cases scanned: `28`
- total passed TF symlinks created: `131`
- summary report:
  - `outputs/overfitting_usage/qc/passed_tf_links_summary.json`

Spot checks confirmed correct link targets, e.g.:

- `qc/batch_0001/s0137/TF02`
- `qc/batch_0002/s1237/TF03`
- `qc/batch_0003/s1103/TF06`
