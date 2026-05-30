#!/usr/bin/env python3
"""
在已生成的「每病例多 TF 子目录」数据集上，按规则筛掉质量差的 <case>/<case>_k 子树。

可用依据：
  1) 父目录 render_qc.json 里每条 TF 的 metrics（nonzero_ratio、mean_intensity、intensity_std）；
  2) points.ply 里 opacity 列为 3DGS 预激活（logit），脚本会算 sigmoid 后的线性不透明度统计。

自动筛选：至少指定一种阈值（如 --max-nonzero-ratio），或 --path-glob；命中规则为「任一条件满足即命中」（OR）。

--json-out：写出命中清单（及可选未命中摘要），便于人工核对后再 --apply。

默认仅打印到 stderr；需显式 --apply 才移动目录到数据集根下 _quarantine/。

示例（自动筛 + 写 JSON）:
  python scripts/prune_multi_tf_case_variants.py \\
    --dataset-root outputs/high_quality_medical_top20_random_multi_tf \\
    --max-nonzero-ratio 0.5 \\
    --json-out outputs/prune_auto_hits.json

示例（通配 + JSON）:
  python scripts/prune_multi_tf_case_variants.py \\
    --dataset-root outputs/high_quality_medical_top20_random_multi_tf \\
    --path-glob 's0001/s0001_[2-7]' \\
    --json-out outputs/prune_glob_hits.json

示例（按病例内相对「最差 K 个」— 避免全库 nonzero_ratio 一刀切的误判）:
  python scripts/prune_multi_tf_case_variants.py \\
    --dataset-root outputs/high_quality_medical_top20_random_multi_tf \\
    --strategy per-case-worst-k --worst-k 2 --rank-by fog \\
    --json-out outputs/prune_per_case_fog_top2.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _read_ply_opacity_stats(ply: Path) -> dict[str, float] | None:
    if not ply.is_file():
        return None
    text = ply.read_text(encoding="utf-8", errors="replace").splitlines()
    if "end_header" not in text:
        return None
    hdr_i = text.index("end_header")
    props: list[str] = []
    for L in text[:hdr_i]:
        if L.startswith("property float "):
            props.append(L.split()[-1])
    if "opacity" not in props:
        return None
    oi = props.index("opacity")
    logits: list[float] = []
    for L in text[hdr_i + 1 : hdr_i + 1 + 200_000]:
        if not L.strip():
            continue
        logits.append(float(L.split()[oi]))
    o = np.asarray(logits, dtype=np.float64)
    lin = 1.0 / (1.0 + np.exp(-np.clip(o, -50, 50)))
    return {
        "n_points": float(lin.size),
        "mean_linear_opacity": float(lin.mean()),
        "p95_linear_opacity": float(np.percentile(lin, 95)),
        "frac_linear_opacity_gt_0.5": float(np.mean(lin > 0.5)),
        "frac_linear_opacity_gt_0.8": float(np.mean(lin > 0.8)),
    }


def _load_qc_index(case_dir: Path) -> dict[str, dict]:
    qc = case_dir / "render_qc.json"
    if not qc.is_file():
        return {}
    data = json.loads(qc.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for it in data.get("items") or []:
        name = it.get("tf_name")
        if isinstance(name, str):
            out[name] = it.get("metrics") or {}
    return out


def _collect_variant_dirs(case_dir: Path) -> list[Path]:
    cid = case_dir.name
    out: list[Path] = []
    for p in sorted(case_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "_quarantine":
            continue
        if re.fullmatch(re.escape(cid) + r"_\d+", p.name):
            out.append(p)
    return out


def _qc_rank_score(metrics: dict[str, Any], rank_by: str) -> float | None:
    nz = metrics.get("nonzero_ratio")
    if nz is None:
        return None
    nz_f = float(nz)
    if rank_by == "nonzero_ratio":
        return nz_f
    std = metrics.get("intensity_std")
    if rank_by == "fog":
        if std is None:
            return nz_f
        return nz_f / (float(std) + 1e-6)
    raise ValueError(f"unknown rank_by: {rank_by}")


def _needs_ply_stats(args: argparse.Namespace) -> bool:
    return any(
        x is not None
        for x in (args.max_mean_linear_opacity, args.min_frac_lin_gt_half, args.max_frac_lin_gt_half)
    )


def _reasons_for_variant(
    vdir: Path,
    glob_paths: set[Path],
    args: argparse.Namespace,
    qc_by_tf: dict[str, dict],
    st: dict[str, float] | None,
) -> list[str]:
    reason: list[str] = []
    if vdir in glob_paths:
        reason.append("path_glob")
    m_qc = qc_by_tf.get(vdir.name) or {}
    nz = m_qc.get("nonzero_ratio")
    mi = m_qc.get("mean_intensity")
    if args.max_nonzero_ratio is not None and nz is not None and float(nz) > args.max_nonzero_ratio:
        reason.append(f"nonzero_ratio={float(nz):.6g}>{args.max_nonzero_ratio}")
    if args.min_mean_intensity is not None and mi is not None and float(mi) < args.min_mean_intensity:
        reason.append(f"mean_intensity={float(mi):.6g}<{args.min_mean_intensity}")
    if st is not None:
        if args.max_mean_linear_opacity is not None and st["mean_linear_opacity"] > args.max_mean_linear_opacity:
            reason.append(
                f"mean_linear_opacity={st['mean_linear_opacity']:.6g}>{args.max_mean_linear_opacity}"
            )
        fg = st["frac_linear_opacity_gt_0.5"]
        if args.min_frac_lin_gt_half is not None and fg < args.min_frac_lin_gt_half:
            reason.append(f"frac_lin_gt_0.5={fg:.6g}<{args.min_frac_lin_gt_half}")
        if args.max_frac_lin_gt_half is not None and fg > args.max_frac_lin_gt_half:
            reason.append(f"frac_lin_gt_0.5={fg:.6g}>{args.max_frac_lin_gt_half}")
    return reason


def _hit_dict(root: Path, vdir: Path, reasons: list[str], st: dict[str, float] | None, qc_by_tf: dict[str, dict]) -> dict[str, Any]:
    m = qc_by_tf.get(vdir.name) or {}
    return {
        "path_relative": str(vdir.relative_to(root)).replace("\\", "/"),
        "case_id": vdir.parent.name,
        "tf_variant": vdir.name,
        "reasons": reasons,
        "qc_metrics": {k: m.get(k) for k in ("nonzero_ratio", "mean_intensity", "intensity_std", "num_images") if k in m},
        "ply_opacity_stats": st,
    }


def _write_json_report(
    path: Path,
    root: Path,
    criteria: dict[str, Any],
    hits: list[dict[str, Any]],
    passed: list[dict[str, Any]] | None,
    total_scanned: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_root": str(root),
        "criteria": criteria,
        "total_variants_scanned": total_scanned,
        "total_hits": len(hits),
        "hits": hits,
        "note": "global_threshold：命中为「任一规则满足」(OR)。per-case-worst-k：每个病例内独立排名，与全库刻度无关。",
    }
    if passed is not None:
        payload["passed"] = passed
        payload["total_passed"] = len(passed)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True, help="含 s0001/s0002/... 的根目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不移动（与 --apply 互斥推荐）")
    ap.add_argument("--apply", action="store_true", help="将命中目录移动到 <dataset-root>/_quarantine/…")
    ap.add_argument(
        "--path-glob",
        type=str,
        default=None,
        help="相对 dataset-root 的 glob；命中则记为 path_glob 原因（可与阈值组合）",
    )
    ap.add_argument("--max-nonzero-ratio", type=float, default=None, help="render_qc nonzero_ratio 上限")
    ap.add_argument("--min-mean-intensity", type=float, default=None, help="render_qc mean_intensity 下限")
    ap.add_argument("--max-mean-linear-opacity", type=float, default=None, help="points.ply 线性不透明度均值上限")
    ap.add_argument(
        "--min-frac-high-opacity",
        type=float,
        default=None,
        dest="min_frac_lin_gt_half",
        help="线性不透明度(sigmoid 后) > 0.5 的点的占比下限",
    )
    ap.add_argument(
        "--max-frac-high-opacity",
        type=float,
        default=None,
        dest="max_frac_lin_gt_half",
        help="线性不透明度 > 0.5 的点的占比上限",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="写入筛选报告 JSON（hits 列表 + 使用到的 criteria）",
    )
    ap.add_argument(
        "--json-list-passed",
        action="store_true",
        help="在 JSON 中附带未命中项（默认仅 qc；加 --json-with-ply-stats 才会读每个 passed 的 ply，很慢）",
    )
    ap.add_argument(
        "--json-with-ply-stats",
        action="store_true",
        help="命中/未命中 JSON 记录中写入 ply 不透明度统计（纯 qc 筛选时默认不读 ply 以加速）",
    )
    ap.add_argument(
        "--strategy",
        choices=("global_threshold", "per-case-worst-k"),
        default="global_threshold",
        help="global_threshold=全库统一阈值；per-case-worst-k=每个病例内按 rank-by 排名只取最差 K 个",
    )
    ap.add_argument(
        "--worst-k",
        type=int,
        default=2,
        help="strategy=per-case-worst-k 时，每个病例标记为命中的前 K 个（按差序）",
    )
    ap.add_argument(
        "--rank-by",
        choices=("nonzero_ratio", "fog"),
        default="fog",
        help="per-case-worst-k 的排名分数：nonzero_ratio 越高越差；fog=nz/(intensity_std+eps) 偏高更像「亮而平」的雾",
    )
    args = ap.parse_args()

    root = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    glob_paths: set[Path] = set()
    if args.path_glob:
        for p in root.glob(args.path_glob):
            if p.is_dir():
                glob_paths.add(p.resolve())

    use_thresholds = any(
        x is not None
        for x in (
            args.max_nonzero_ratio,
            args.min_mean_intensity,
            args.max_mean_linear_opacity,
            args.min_frac_lin_gt_half,
            args.max_frac_lin_gt_half,
        )
    )
    if args.strategy == "per-case-worst-k":
        if args.worst_k < 1:
            raise SystemExit("--worst-k 必须 >= 1")
    elif not use_thresholds and not glob_paths:
        raise SystemExit("请至少指定 --path-glob 或一种数值阈值（如 --max-nonzero-ratio），或改用 --strategy per-case-worst-k。")

    criteria = {
        "strategy": args.strategy,
        "worst_k": args.worst_k if args.strategy == "per-case-worst-k" else None,
        "rank_by": args.rank_by if args.strategy == "per-case-worst-k" else None,
        "path_glob": args.path_glob,
        "max_nonzero_ratio": args.max_nonzero_ratio,
        "min_mean_intensity": args.min_mean_intensity,
        "max_mean_linear_opacity": args.max_mean_linear_opacity,
        "min_frac_high_opacity": args.min_frac_lin_gt_half,
        "max_frac_high_opacity": args.max_frac_lin_gt_half,
    }

    only_glob = bool(glob_paths) and not use_thresholds and args.strategy == "global_threshold"

    quarantine = root / "_quarantine"
    hits: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    to_move: list[tuple[Path, str]] = []
    total_scanned = 0

    if args.strategy == "per-case-worst-k":
        need_ply = _needs_ply_stats(args)
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("s")):
            if case_dir.name == "_quarantine":
                continue
            qc_by_tf = _load_qc_index(case_dir)
            scored: list[tuple[float, Path, dict[str, Any]]] = []
            for vdir in _collect_variant_dirs(case_dir):
                total_scanned += 1
                m = qc_by_tf.get(vdir.name) or {}
                sc = _qc_rank_score(m, args.rank_by)
                if sc is None:
                    continue
                scored.append((sc, vdir, m))
            scored.sort(key=lambda t: -t[0])
            worst = scored[: min(args.worst_k, len(scored))]
            for rank_idx, (sc, vdir, m) in enumerate(worst):
                reasons: list[str] = [
                    f"per_case_worst_k={args.worst_k} intra_rank={rank_idx + 1} rank_by={args.rank_by} score={sc:.6g}"
                ]
                if vdir in glob_paths:
                    reasons.append("path_glob")
                st = _read_ply_opacity_stats(vdir / "points.ply") if need_ply or args.json_with_ply_stats else None
                rec = _hit_dict(root, vdir, reasons, st, qc_by_tf)
                hits.append(rec)
                to_move.append((vdir, "; ".join(reasons)))
                print(f"HIT {rec['path_relative']} :: {'; '.join(reasons)}", file=sys.stderr)
                if st:
                    print(
                        f"    ply: mean_lin={st['mean_linear_opacity']:.4f} p95={st['p95_linear_opacity']:.4f} frac>0.5={st['frac_linear_opacity_gt_0.5']:.4f}",
                        file=sys.stderr,
                    )
                print(
                    f"    qc:  nonzero_ratio={m.get('nonzero_ratio')} intensity_std={m.get('intensity_std')} mean_intensity={m.get('mean_intensity')}",
                    file=sys.stderr,
                )
            if args.json_list_passed:
                for sc, vdir, m in scored[len(worst) :]:
                    pst = _read_ply_opacity_stats(vdir / "points.ply") if args.json_with_ply_stats else None
                    passed.append(
                        {
                            "path_relative": str(vdir.relative_to(root)).replace("\\", "/"),
                            "case_id": vdir.parent.name,
                            "tf_variant": vdir.name,
                            "rank_score": sc,
                            "qc_metrics": {k: m.get(k) for k in ("nonzero_ratio", "mean_intensity", "intensity_std") if k in m},
                            "ply_opacity_stats": pst,
                        }
                    )
        for vdir in sorted(glob_paths, key=lambda p: str(p)):
            if not vdir.is_dir() or vdir in {x[0] for x in to_move}:
                continue
            total_scanned += 1
            case_dir = vdir.parent
            qc_by_tf = _load_qc_index(case_dir)
            st = _read_ply_opacity_stats(vdir / "points.ply") if need_ply or args.json_with_ply_stats else None
            reasons = ["path_glob_only"]
            rec = _hit_dict(root, vdir, reasons, st, qc_by_tf)
            hits.append(rec)
            to_move.append((vdir, "; ".join(reasons)))
            print(f"HIT {rec['path_relative']} :: path_glob_only", file=sys.stderr)
    elif only_glob:
        for vdir in sorted(glob_paths, key=lambda p: str(p)):
            if not vdir.is_dir():
                continue
            total_scanned += 1
            case_dir = vdir.parent
            qc_by_tf = _load_qc_index(case_dir)
            st = _read_ply_opacity_stats(vdir / "points.ply")
            reasons = _reasons_for_variant(vdir, glob_paths, args, qc_by_tf, st)
            rec = _hit_dict(root, vdir, reasons, st, qc_by_tf)
            hits.append(rec)
            to_move.append((vdir, "; ".join(reasons)))
            print(f"HIT {rec['path_relative']} :: {'; '.join(reasons)}", file=sys.stderr)
            if st:
                print(
                    f"    ply: mean_lin={st['mean_linear_opacity']:.4f} p95={st['p95_linear_opacity']:.4f} frac>0.5={st['frac_linear_opacity_gt_0.5']:.4f}",
                    file=sys.stderr,
                )
            m = qc_by_tf.get(vdir.name) or {}
            print(f"    qc:  nonzero_ratio={m.get('nonzero_ratio')} mean_intensity={m.get('mean_intensity')}", file=sys.stderr)
    else:
        need_ply = _needs_ply_stats(args)
        for case_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("s")):
            if case_dir.name == "_quarantine":
                continue
            qc_by_tf = _load_qc_index(case_dir)
            for vdir in _collect_variant_dirs(case_dir):
                total_scanned += 1
                st = _read_ply_opacity_stats(vdir / "points.ply") if need_ply else None
                reasons = _reasons_for_variant(vdir, glob_paths, args, qc_by_tf, st)
                if reasons and args.json_with_ply_stats and st is None:
                    st = _read_ply_opacity_stats(vdir / "points.ply")
                if reasons:
                    rec = _hit_dict(root, vdir, reasons, st, qc_by_tf)
                    hits.append(rec)
                    to_move.append((vdir, "; ".join(reasons)))
                    print(f"HIT {rec['path_relative']} :: {'; '.join(reasons)}", file=sys.stderr)
                    if st:
                        print(
                            f"    ply: mean_lin={st['mean_linear_opacity']:.4f} p95={st['p95_linear_opacity']:.4f} frac>0.5={st['frac_linear_opacity_gt_0.5']:.4f}",
                            file=sys.stderr,
                        )
                    m = qc_by_tf.get(vdir.name) or {}
                    print(f"    qc:  nonzero_ratio={m.get('nonzero_ratio')} mean_intensity={m.get('mean_intensity')}", file=sys.stderr)
                elif args.json_list_passed:
                    m = qc_by_tf.get(vdir.name) or {}
                    pst = st
                    if args.json_with_ply_stats and pst is None:
                        pst = _read_ply_opacity_stats(vdir / "points.ply")
                    passed.append(
                        {
                            "path_relative": str(vdir.relative_to(root)).replace("\\", "/"),
                            "case_id": vdir.parent.name,
                            "tf_variant": vdir.name,
                            "qc_metrics": {k: m.get(k) for k in ("nonzero_ratio", "mean_intensity", "intensity_std") if k in m},
                            "ply_opacity_stats": pst,
                        }
                    )

    print(f"total_scanned={total_scanned} total_hit={len(to_move)}", file=sys.stderr)

    json_passed = passed if args.json_list_passed else None
    if args.json_out:
        outp = args.json_out if args.json_out.is_absolute() else (Path.cwd() / args.json_out)
        _write_json_report(outp, root, criteria, hits, json_passed, total_scanned)

    if not to_move:
        if not args.json_out:
            print("无命中项。", file=sys.stderr)
        return

    if args.apply:
        if args.dry_run:
            print("同时给了 --apply 与 --dry-run：不执行移动。", file=sys.stderr)
            return
        quarantine.mkdir(parents=True, exist_ok=True)
        for vdir, _why in to_move:
            rel = vdir.relative_to(root)
            dst = quarantine / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(vdir), str(dst))
            print(f"moved -> {dst}", file=sys.stderr)
    elif not args.apply:
        print("未指定 --apply：未移动目录。确认 JSON 后追加 --apply 执行隔离。", file=sys.stderr)


if __name__ == "__main__":
    main()
