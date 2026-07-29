"""Per-cell uncertainty + paired factor contrasts for ablation discriminability."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_bench.config_load import DOCS_DIR, RESULTS_DIR, ensure_dirs

# Pre-registered key factors for primary claims
KEY_FACTORS = [
    "retrieval",
    "rerank",
    "chunk_strategy",
    "top_k",
    "threshold",
    "embeddings",
]


def proportion_se(p_hat: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = min(max(float(p_hat), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n)


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = min(max(float(p_hat), 0.0), 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def cell_uncertainty(
    n: int,
    p_hat: float | None = None,
    hits: int | None = None,
    hit_vector: Sequence[int] | None = None,
) -> dict[str, Any]:
    if hit_vector is not None:
        n = len(hit_vector)
        hits = int(sum(hit_vector))
        p_hat = hits / n if n else 0.0
    elif hits is not None and n:
        p_hat = hits / n
    elif p_hat is None:
        p_hat = 0.0
    se = proportion_se(float(p_hat), n)
    lo, hi = wilson_ci(float(p_hat), n)
    return {
        "n": n,
        "p_hat": float(p_hat),
        "se": se,
        "wilson_95": [lo, hi],
        "hits": hits if hits is not None else int(round(float(p_hat) * n)),
    }


def paired_delta(
    hits_a: Sequence[int],
    hits_b: Sequence[int],
) -> dict[str, Any]:
    """Paired mean(hit_a - hit_b) with SE and whether 95% CI excludes 0."""
    a = list(hits_a)
    b = list(hits_b)
    if len(a) != len(b) or not a:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "se_paired": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "discriminative": False,
            "reason": "length_mismatch_or_empty",
        }
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    if n > 1:
        var = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    if se == 0.0 or (isinstance(se, float) and se != se):  # nan
        # Perfect agreement or single obs
        disc = abs(mean_d) > 1e-12
        ci = [mean_d, mean_d] if se == 0.0 else [float("nan"), float("nan")]
    else:
        z = 1.96
        ci = [mean_d - z * se, mean_d + z * se]
        disc = (ci[0] > 0.0) or (ci[1] < 0.0) or (abs(mean_d) > 1.96 * se)
    # McNemar-style discordant counts
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    return {
        "n": n,
        "mean_delta": mean_d,
        "se_paired": se,
        "ci_95": ci,
        "discriminative": bool(disc),
        "mcnemar_a_only": a_only,
        "mcnemar_b_only": b_only,
    }


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def _norm_val(v: Any) -> str:
    if v is None or v == "":
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    factors: Sequence[str] | None = None,
    label: str = "run",
) -> dict[str, Any]:
    """
    rows: ablation rows with recall_at_k, n, and optionally hit_vector (list[int]).
    """
    factors = list(factors or KEY_FACTORS)
    cells = []
    for r in rows:
        n = int(r.get("n") or 0)
        p = float(r.get("recall_at_k") or 0.0)
        hv = r.get("hit_vector")
        unc = cell_uncertainty(n, p_hat=p, hit_vector=hv if isinstance(hv, list) else None)
        cells.append(
            {
                "cell_id": r.get("cell_id"),
                "config": {
                    k: r.get(k)
                    for k in (
                        "retrieval",
                        "rerank",
                        "chunk_strategy",
                        "top_k",
                        "threshold",
                        "embeddings",
                        "retriever",
                    )
                    if k in r
                },
                "uncertainty": unc,
                "has_hit_vector": isinstance(hv, list),
            }
        )

    factor_results: dict[str, Any] = {}
    for fac in factors:
        if fac not in (rows[0] if rows else {}):
            # still allow embeddings etc. if present on some rows
            if not any(fac in r for r in rows):
                continue
        # For non-retrieval factors, restrict to retrieval=true cells
        if fac == "retrieval":
            subset = rows
        else:
            subset = [r for r in rows if _as_bool(r.get("retrieval", True))]
        levels: dict[str, list[dict]] = {}
        for r in subset:
            if fac not in r:
                continue
            levels.setdefault(_norm_val(r.get(fac)), []).append(r)

        level_stats = {}
        for lv, rs in levels.items():
            recalls = [float(x.get("recall_at_k") or 0.0) for x in rs]
            mean_r = sum(recalls) / len(recalls) if recalls else 0.0
            # Prefer pooled hit vector if all cells share same qids length — use best cell's vector
            best = max(rs, key=lambda x: float(x.get("recall_at_k") or 0.0))
            hv = best.get("hit_vector")
            n_q = int(best.get("n") or 0)
            unc = cell_uncertainty(
                n_q,
                p_hat=float(best.get("recall_at_k") or 0.0),
                hit_vector=hv if isinstance(hv, list) else None,
            )
            level_stats[lv] = {
                "n_cells": len(rs),
                "mean_recall": mean_r,
                "best_cell_id": best.get("cell_id"),
                "best_uncertainty": unc,
            }

        # Pairwise paired contrasts across levels using hit vectors when available
        level_names = sorted(levels.keys())
        pairwise = []
        for i, a_name in enumerate(level_names):
            for b_name in level_names[i + 1 :]:
                # Choose representative cells with same other factors when possible:
                # pair best cell of each level if hit vectors exist and same length
                a_best = max(levels[a_name], key=lambda x: float(x.get("recall_at_k") or 0.0))
                b_best = max(levels[b_name], key=lambda x: float(x.get("recall_at_k") or 0.0))
                ha, hb = a_best.get("hit_vector"), b_best.get("hit_vector")
                if isinstance(ha, list) and isinstance(hb, list) and len(ha) == len(hb):
                    pd = paired_delta(ha, hb)
                    method = "paired_hit_vector"
                else:
                    # Fallback: unpaired difference of means with pooled SE (cell-level)
                    ra = [float(x.get("recall_at_k") or 0.0) for x in levels[a_name]]
                    rb = [float(x.get("recall_at_k") or 0.0) for x in levels[b_name]]
                    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
                    # SE of difference of independent means (approx using query-level n from first)
                    n_ref = int(a_best.get("n") or 35)
                    se_a = proportion_se(ma, n_ref)
                    se_b = proportion_se(mb, n_ref)
                    se_d = math.sqrt(se_a**2 + se_b**2)
                    mean_d = ma - mb
                    ci = [mean_d - 1.96 * se_d, mean_d + 1.96 * se_d]
                    disc = (ci[0] > 0) or (ci[1] < 0)
                    pd = {
                        "n": n_ref,
                        "mean_delta": mean_d,
                        "se_paired": se_d,
                        "ci_95": ci,
                        "discriminative": disc,
                        "note": "unpaired_cell_means_fallback",
                    }
                    method = "unpaired_means"
                pairwise.append(
                    {
                        "level_a": a_name,
                        "level_b": b_name,
                        "method": method,
                        **pd,
                    }
                )

        any_disc = any(p.get("discriminative") for p in pairwise)
        factor_results[fac] = {
            "levels": level_stats,
            "pairwise": pairwise,
            "discriminative": bool(any_disc),
            "claim_label": "discriminative" if any_disc else "non-discriminative",
        }

    return {
        "label": label,
        "n_cells": len(rows),
        "cells": cells,
        "factors": factor_results,
        "key_factors": list(factors),
    }


def write_discriminability(
    analysis: dict[str, Any],
    path: Path | None = None,
) -> Path:
    ensure_dirs()
    path = path or (RESULTS_DIR / "discriminability.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def write_discriminability_md(
    analysis: dict[str, Any],
    path: Path | None = None,
) -> Path:
    ensure_dirs()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (DOCS_DIR / "discriminability.md")
    lines = [
        "# Discriminability report",
        "",
        f"Label set / run: `{analysis.get('label')}`",
        f"Cells: {analysis.get('n_cells')}",
        "",
        "Per-cell Wilson 95% CI and paired factor contrasts.",
        "A factor is **discriminative** if any pairwise 95% CI on Δ excludes 0.",
        "",
        "## Key factors",
        "",
        "| factor | claim | n_levels |",
        "| --- | --- | ---: |",
    ]
    factors = analysis.get("factors") or {}
    for fac, fr in factors.items():
        n_lv = len(fr.get("levels") or {})
        lines.append(f"| {fac} | {fr.get('claim_label')} | {n_lv} |")
    lines.append("")
    for fac, fr in factors.items():
        lines.append(f"### {fac} — {fr.get('claim_label')}")
        lines.append("")
        for pair in fr.get("pairwise") or []:
            disc = "YES" if pair.get("discriminative") else "no"
            lines.append(
                f"- `{pair.get('level_a')}` vs `{pair.get('level_b')}`: "
                f"Δ={pair.get('mean_delta'):.4f} "
                f"CI={pair.get('ci_95')} "
                f"method={pair.get('method')} "
                f"discriminative={disc}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_rows_from_csv(csv_path: Path) -> list[dict[str, Any]]:
    import csv

    rows = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            row: dict[str, Any] = dict(r)
            row["cell_id"] = int(r["cell_id"]) if r.get("cell_id", "").isdigit() else i
            row["recall_at_k"] = float(r.get("recall_at_k") or 0)
            row["attribution_rate"] = float(r.get("attribution_rate") or 0)
            row["n"] = int(float(r.get("n") or 0))
            # booleans
            if "retrieval" in r:
                row["retrieval"] = _as_bool(r["retrieval"])
            if "rerank" in r:
                row["rerank"] = _as_bool(r["rerank"])
            if "top_k" in r and r["top_k"] != "":
                row["top_k"] = int(float(r["top_k"]))
            thr = r.get("threshold", "")
            row["threshold"] = None if thr in ("", "null", "None") else float(thr)
            rows.append(row)
    return rows
