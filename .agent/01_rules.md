# 01_RULES.md

> This file defines the execution rules for agent / Codex.
> It does not explain the project itself.
> Before editing code, the agent should read both `00_CONTEXT.md` and this file.

---

## 1. Pre-Edit Rules

Before making any code change, the agent must first understand:

1. target: what module is being changed, and what role it plays
2. behavior: whether the change is implementation-level or behavior-changing
3. scope: what upstream, downstream, and experiment-level impact may follow
4. design quality: whether the change preserves a reasonably general module design

If these questions are not clear, the agent should not make large changes directly.

If these are not clear, the agent should not make large changes directly.

### Required steps before editing
1. Read `00_CONTEXT.md`
2. Locate the relevant call chain
3. Classify the change as:
   - implementation-only
   - training-strategy change
   - algorithm-semantics change
4. Estimate the impact scope:
   - local module
   - interface chain
   - training behavior
   - experiment comparability

## 2. Editing Rules
Editing behavior must remain:

1. honest: do not hide semantic errors with implementation tricks
2. consistent: do not break end-to-end consistency through partial changes
3. explicit: keep tensor semantics, interface semantics, and algorithm structure clear in code
4. accountable: high-risk behavior changes must never be made silently

### 2.1 Core implementation principles

1. Do not hide semantic errors with implementation tricks.
   - No silent shape hacks.
   - No unjustified dtype casts.
   - No silent device movement that hides bugs.
   - No temporary reshape used only to make code run.

2. Do not break end-to-end consistency through partial interface changes.
   - Do not change returned field names without updating the full pipeline.
   - Do not change sample keys without updating dataset, trainer, evaluator, and tests.

3. Do not mix temporary or case-specific logic into the stable main path.
   - Experimental logic should not be written directly into the stable production/research path.
   - Temporary fixes should not become permanent hidden behavior.

4. Keep code structure explicit.
   - Keep tensor semantics explicit.
   - Add shape annotations for key tensors.
   - Preserve interface stability when possible.
   - Prefer structured outputs over long tuples.
   - Add sanity checks for shape, dtype, and device when useful.
   - Keep refactor separate from semantic algorithm change when possible.

### 2.2 High-risk changes that must not be silent

### 2.3 Shape annotation rule
Key modules must include explicit tensor shape comments for:

- inputs
- outputs
- reshape / permute / flatten / unflatten steps
- intermediate tensors that are easy to misunderstand

---

## 3. Training Conventions
Training behavior must remain:

1. reproducible: Same config should produce comparable results
2. recoverable: Training code must preserve complete checkpoint behavior.
3. observable: Training logs must be sufficient for debugging and experiment comparison.
4. traceable: Experiment names should be stable, readable, and easy to trace.
5. clean: temporary runs should be deleted immediately after the attempt finishes, unless those outputs are explicitly needed for comparison, debugging, or record keeping.

### 3.1 Failed Experiment Cleanup Rule
The agent must always clean failed or superseded experiments promptly.

Required behavior:
- If an experiment is confirmed wrong, invalid, or no longer needed, delete its outputs promptly.
- If a newer experiment replaces an older one for the same purpose, the older one should be removed unless the user explicitly wants to keep it.
- Cleanup must include:
  - run directories
  - launch logs
  - training logs
  - GPU monitor logs
  - pid files
- Before deletion, the agent must make sure the target is not an active run.
- If the failed experiment produced an important conclusion, keep the conclusion in `02_WORKLOG.md`, but do not keep the full stale outputs by default.

### 3.2 Background Training Rule
All training jobs must be launched in the background by default.

Required behavior:
- Do not keep long-running training attached to the foreground terminal.
- Every training launch must write stdout/stderr to a persistent log file.
- Every training launch must record enough metadata to find the run again:
  - config path
  - run/output directory
  - log file path
  - pid when available
- After launch, the agent must verify that the process is still alive instead of assuming the background start succeeded.
- If resuming training from a checkpoint, the resume source must be stated explicitly in the launch record.

Preferred practice:
- Launch with a background-safe pattern such as `nohup ... > logfile 2>&1 &`.
- Use a readable timestamped log filename under `outputs/` or the corresponding run directory.
- Report the launch command summary and where to monitor the log.

Exception:
- Foreground execution is only acceptable for short smoke tests whose purpose is to validate startup, imports, or immediate crash behavior.
- If a foreground smoke test is used, the real training run must still be relaunched in the background afterward.

## 4. Experiment Result Recording

Meaningful experiment results and analysis must be recorded in `02_WORKLOG.md`.

`02_WORKLOG.md` is intended for concise high-level experiment logs, not raw terminal dumps or low-level debugging traces.

Each meaningful entry should record, when applicable:

- experiment goal
- key changes
- observed results
- analysis and interpretation
- lessons learned or conclusions
- what was validated as correct
- what was shown to be incorrect
- open questions
- next-step suggestions

Important:
- `01_RULES.md` defines rules and expectations
- `02_WORKLOG.md` records concise experiment outcomes, analysis, and conclusions


## 5. Color Conventions
- Convention 体积渲染调用 `pvpython` 时，不要预乘 alpha；输出的 PNG / RGBA 结果应保持 straight alpha，而不是 premultiplied alpha。
- 渲染阶段当 transfer function 作用到物理场并生成 RGBA volume 时，`rgb` 不要乘以 `alpha`。
- 采样阶段如果需要基于 transfer function 生成 RGBA volume，同样不要做 `rgb *= alpha`。
- 新增或修改与 transfer function、RGBA volume、PNG 导出、opacity sampling 相关的代码时，默认遵循“非预乘 alpha”约定；如果必须偏离，先显式说明原因并同步更新规则。