#!/usr/bin/env python3
"""
离线从 raw/totalseg3d 里选「CT 标量统计上」贴近测试病例的训练 case。

思路（与 docs/totalseg3d_tf_reuse_and_expand_cn.md 第 10 节一致）：
  1. 对测试集各例的 ct.nii.gz 做同一套随机子采样；
  2. 在子采样体素上算低维特征：mean/std/分位数 + 固定 HU 轴上的归一化直方图；
  3. 聚合方式（--aggregate）：
     - mean-prototype：对测试各例特征取平均得原型，再算距离（测试内部差异大时可能被「抹平」）；
     - min-to-test：对每个候选，算到「每一个」测试例特征的距离，取最小值（贴近某一测试例即可）。
  4. 选法（--select-mode）：
     - global：按得分全局排序取 Top-K（min-to-test 时可能全落在少数「好碰」的测试锚点上）；
     - balanced-anchors：仅 min-to-test 可用，按最近锚点分桶后对每个测试例近似均分名额，再不足则用全局递补。

依赖：pip install nibabel  或  pip install -e ".[medical]"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _require_nibabel():
    try:
        import nibabel as nib  # noqa: F401

        return nib
    except ImportError:
        print("需要 nibabel：pip install nibabel  或  pip install 'vol2splat[medical]'", file=sys.stderr)
        raise


def _subsample_flat(data: np.ndarray, n_cap: int, rng: np.random.Generator) -> np.ndarray:
    flat = np.asarray(data, dtype=np.float32).ravel()
    if flat.size == 0:
        return flat
    if flat.size <= n_cap:
        return flat
    idx = rng.choice(flat.size, size=n_cap, replace=False)
    return flat[idx]


def _feature_from_sample(
    samp: np.ndarray,
    hist_edges: np.ndarray,
) -> np.ndarray:
    """samp: 1D float32 子采样体素。"""
    assert samp.ndim == 1
    mean = float(np.mean(samp))
    std = float(np.std(samp))
    p1, p50, p99 = (float(x) for x in np.percentile(samp, (1.0, 50.0, 99.0)))
    counts, _ = np.histogram(samp, bins=hist_edges)
    total = float(np.sum(counts)) + 1e-9
    hist = counts.astype(np.float64) / total
    return np.array([mean, std, p1, p50, p99, *hist.tolist()], dtype=np.float64)


def _load_ct_path(root: Path, case_id: str) -> Path:
    p = root / case_id / "ct.nii.gz"
    if not p.is_file():
        raise FileNotFoundError(f"missing ct: {p}")
    return p


def _case_feature(
    nib,
    root: Path,
    case_id: str,
    n_subsample: int,
    hist_edges: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    path = _load_ct_path(root, case_id)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    samp = _subsample_flat(data, n_cap=n_subsample, rng=rng)
    return _feature_from_sample(samp, hist_edges=hist_edges)


def main() -> None:
    nib = _require_nibabel()
    ap = argparse.ArgumentParser(description="按 CT 子采样统计贴近测试集，挑选训练 case_id。")
    ap.add_argument(
        "--totalseg-root",
        type=Path,
        default=Path("raw/totalseg3d"),
        help="含 <case_id>/ct.nii.gz 的根目录（相对 cwd 或绝对路径）",
    )
    ap.add_argument(
        "--test-case-ids",
        type=str,
        default="s0244,s0652,s0693,s0703,s1009,s1073,s1348",
        help="逗号分隔，与 batch_high_quality_medical 测试集一致",
    )
    ap.add_argument("--top-k", type=int, default=20, help="输出距离最小的 K 个训练候选")
    ap.add_argument("--seed", type=int, default=0, help="子采样与候选打乱用种子（固定可复现）")
    ap.add_argument(
        "--n-subsample",
        type=int,
        default=200_000,
        help="每例从体数据中随机采样的体素数（越大越稳、越慢）",
    )
    ap.add_argument(
        "--hist-min",
        type=float,
        default=-1024.0,
        help="直方图左端（HU 轴固定，便于跨病例可比）",
    )
    ap.add_argument(
        "--hist-max",
        type=float,
        default=3071.0,
        help="直方图右端",
    )
    ap.add_argument("--hist-bins", type=int, default=16, help="直方图 bin 数（特征维 +5）")
    ap.add_argument(
        "--aggregate",
        choices=("mean-prototype", "min-to-test"),
        default="mean-prototype",
        help="mean-prototype=到测试特征均值的距离；min-to-test=到「最近一条」测试例的最小距离（测试集内部差异大时更合理）",
    )
    ap.add_argument(
        "--metric",
        choices=("l2", "l1"),
        default="l2",
        help="特征空间中的 L2 或 L1（与 --aggregate 组合使用）",
    )
    ap.add_argument(
        "--scale-by-pool",
        action="store_true",
        help="对特征各维用「候选池」的 std 做缩放，减轻量纲差异（推荐）",
    )
    ap.add_argument(
        "--select-mode",
        choices=("global", "balanced-anchors"),
        default="global",
        help="global=按分数全局 Top-K；balanced-anchors=按最近测试锚点分桶后均分名额（仅 aggregate=min-to-test）",
    )
    ap.add_argument("--json-out", type=Path, default=None, help="可选：写入完整排序与特征路径")
    ap.add_argument(
        "--score-pool-size",
        type=int,
        default=None,
        help="若设置：从候选池中无放回随机抽这么多个 case 参与打分（其余不扫）；省时间用，默认扫全候选池",
    )
    args = ap.parse_args()
    if args.select_mode == "balanced-anchors" and args.aggregate != "min-to-test":
        raise SystemExit("--select-mode balanced-anchors 仅支持与 --aggregate min-to-test 联用")

    test_ids = tuple(x.strip() for x in args.test_case_ids.split(",") if x.strip())
    test_set = frozenset(test_ids)
    root: Path = args.totalseg_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"totalseg root not found: {root}")

    all_ids = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("s"))
    pool_full = [c for c in all_ids if c not in test_set]
    if not pool_full:
        raise SystemExit("no candidate cases after excluding test ids")

    rng = np.random.default_rng(args.seed)
    pool = pool_full
    pool_was_capped = False
    if args.score_pool_size is not None:
        m = int(args.score_pool_size)
        if m < 1:
            raise SystemExit("--score-pool-size must be >= 1")
        if m < len(pool):
            pick = rng.choice(len(pool), size=m, replace=False)
            pool = sorted(pool[i] for i in pick)
            pool_was_capped = True
    hist_edges = np.linspace(args.hist_min, args.hist_max, args.hist_bins + 1, dtype=np.float64)

    print(f"totalseg_root={root}", file=sys.stderr)
    cap = f"  (random cap={args.score_pool_size}, full_pool={len(pool_full)})" if pool_was_capped else ""
    print(
        f"test n={len(test_ids)}  scored_pool n={len(pool)}  subsample={args.n_subsample}  aggregate={args.aggregate}  select_mode={args.select_mode}{cap}",
        file=sys.stderr,
    )

    test_feats: list[np.ndarray] = []
    for cid in test_ids:
        f = _case_feature(nib, root, cid, n_subsample=args.n_subsample, hist_edges=hist_edges, rng=rng)
        test_feats.append(f)
    test_stack = np.stack(test_feats, axis=0)
    prototype = np.mean(test_stack, axis=0)

    pool_feats: list[tuple[str, np.ndarray]] = []
    for i, cid in enumerate(pool):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  pool {i+1}/{len(pool)} {cid}", file=sys.stderr)
        f = _case_feature(nib, root, cid, n_subsample=args.n_subsample, hist_edges=hist_edges, rng=rng)
        pool_feats.append((cid, f))

    feat_mat = np.stack([f for _, f in pool_feats], axis=0)
    if args.scale_by_pool:
        std = feat_mat.std(axis=0) + 1e-9
        feat_mat_n = feat_mat / std
        test_stack_n = test_stack / std
        prototype_n = prototype / std
    else:
        feat_mat_n = feat_mat
        test_stack_n = test_stack
        prototype_n = prototype

    d_ts: np.ndarray | None = None
    nearest_idx_arr: np.ndarray | None = None
    if args.aggregate == "mean-prototype":
        if args.metric == "l2":
            dists = np.linalg.norm(feat_mat_n - prototype_n.reshape(1, -1), axis=1)
        else:
            dists = np.sum(np.abs(feat_mat_n - prototype_n.reshape(1, -1)), axis=1)
        nearest_test: list[str | None] = [None] * len(dists)
    else:
        diff = feat_mat_n[:, np.newaxis, :] - test_stack_n[np.newaxis, :, :]
        if args.metric == "l2":
            d_ts = np.linalg.norm(diff, axis=2)
        else:
            d_ts = np.sum(np.abs(diff), axis=2)
        nearest_idx_arr = np.argmin(d_ts, axis=1)
        dists = d_ts[np.arange(d_ts.shape[0]), nearest_idx_arr]
        nearest_test = [test_ids[int(i)] for i in nearest_idx_arr]

    order = np.argsort(dists)

    def _balanced_anchor_indices(nearest_idx_arr: np.ndarray) -> list[int]:
        """按「最近测试锚点」分桶；每桶内按到该锚点的距离升序；各锚点近似均分 K 个名额。"""
        assert d_ts is not None
        n_test = len(test_ids)
        n_pool = d_ts.shape[0]
        k = min(args.top_k, n_pool)
        base, rem = divmod(k, n_test)
        quotas = [base + (1 if i < rem else 0) for i in range(n_test)]
        buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_test)]
        for j in range(n_pool):
            ti = int(nearest_idx_arr[j])
            buckets[ti].append((float(d_ts[j, ti]), j))
        for ti in range(n_test):
            buckets[ti].sort(key=lambda x: x[0])
        selected: set[int] = set()
        out: list[int] = []
        ptr = [0] * n_test
        picked = [0] * n_test

        def one_round() -> bool:
            progressed = False
            for ti in range(n_test):
                if len(out) >= k:
                    break
                if picked[ti] >= quotas[ti]:
                    continue
                while ptr[ti] < len(buckets[ti]):
                    _d, j = buckets[ti][ptr[ti]]
                    ptr[ti] += 1
                    if j in selected:
                        continue
                    selected.add(j)
                    out.append(j)
                    picked[ti] += 1
                    progressed = True
                    break
            return progressed

        while len(out) < k and one_round():
            pass
        if len(out) < k:
            for j in order:
                if len(out) >= k:
                    break
                if j in selected:
                    continue
                selected.add(j)
                out.append(int(j))
        return out

    top_k = min(args.top_k, len(order))
    if args.select_mode == "balanced-anchors":
        assert nearest_idx_arr is not None
        sel_idx = _balanced_anchor_indices(nearest_idx_arr)
    else:
        sel_idx = list(order[:top_k])

    if args.aggregate == "min-to-test" and sel_idx:
        c = Counter(str(nearest_test[j]) for j in sel_idx)
        print(f"# 本次输出 {len(sel_idx)} 例的 nearest_test 计数: {dict(c)}", file=sys.stderr)

    title = (
        "与测试特征均值（原型）距离最小"
        if args.aggregate == "mean-prototype"
        else (
            "与「某一」测试例距离最小且按测试锚点均衡配额（balanced-anchors）"
            if args.select_mode == "balanced-anchors"
            else "与「某一」测试例特征距离最小（min-to-test, global Top-K）"
        )
    )
    print(f"\n# {title}的 case_id（可直接贴 yaml batch.case_ids）\n")
    for rank, j in enumerate(sel_idx, start=1):
        cid = pool_feats[j][0]
        nt = nearest_test[j]
        extra = f"  nearest_test={nt}" if nt is not None else ""
        print(f"    - {cid}  # rank={rank}  {args.metric}={float(dists[j]):.6g}{extra}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for rank, j in enumerate(order):
            cid = pool_feats[j][0]
            row = {"rank": rank + 1, "case_id": cid, "distance": float(dists[j])}
            if nearest_test[j] is not None:
                row["nearest_test_id"] = nearest_test[j]
            rows.append(row)
        out_rows = []
        for rank, j in enumerate(sel_idx, start=1):
            cid = pool_feats[j][0]
            row = {"output_rank": rank, "case_id": cid, "distance": float(dists[j])}
            if nearest_test[j] is not None:
                row["nearest_test_id"] = nearest_test[j]
            out_rows.append(row)
        payload = {
            "totalseg_root": str(root),
            "test_case_ids": list(test_ids),
            "aggregate": args.aggregate,
            "select_mode": args.select_mode,
            "seed": args.seed,
            "n_subsample": args.n_subsample,
            "hist_edges": [float(hist_edges[0]), float(hist_edges[-1]), args.hist_bins],
            "metric": args.metric,
            "scale_by_pool": args.scale_by_pool,
            "prototype_mean": prototype.tolist(),
            "ranked_full_pool": rows,
            "output_top_k": out_rows,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n# wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
