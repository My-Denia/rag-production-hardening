"""Build quality_report.json with required schema keys."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs
from rag_bench.selection import rules_content_hash


def _rss_mb() -> float | None:
    try:
        import resource  # type: ignore

        # Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _index_bytes_estimate() -> int:
    cache = Path(__file__).resolve().parents[2] / ".cache"
    total = 0
    if cache.exists():
        for p in cache.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def build_quality_report(
    *,
    by_arm: dict[str, Any],
    selection: dict[str, Any],
    holdout_confirmation: dict[str, Any] | None = None,
    regression_metrics: dict[str, Any] | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    lat_p50 = []
    lat_p95 = []
    by_category: dict[str, Any] = {}
    for aid, m in by_arm.items():
        metrics = m.get("metrics") or m
        if metrics.get("latency_p50_ms") is not None:
            lat_p50.append(float(metrics["latency_p50_ms"]))
        if metrics.get("latency_p95_ms") is not None:
            lat_p95.append(float(metrics["latency_p95_ms"]))
        bc = metrics.get("by_category") or {}
        for cat, stats in bc.items():
            slot = by_category.setdefault(cat, {"n": 0, "arms": {}})
            slot["n"] = max(slot["n"], int(stats.get("n") or 0))
            slot["arms"][aid] = stats

    hc = holdout_confirmation or {}
    deltas = {}
    if hc.get("bootstrap"):
        deltas["delta_recall"] = hc["bootstrap"].get("point_delta")
        deltas["ci_low"] = hc["bootstrap"].get("ci_low")
        deltas["ci_high"] = hc["bootstrap"].get("ci_high")

    report = {
        "n": n or sum(int((m.get("metrics") or m).get("n") or 0) for m in list(by_arm.values())[:1]),
        "by_arm": {
            aid: {
                k: (m.get("metrics") or m).get(k)
                for k in (
                    "recall_at_k",
                    "recall_at_1",
                    "mrr",
                    "attribution_rate",
                    "unsupported_answer_rate",
                    "error_citation_rate",
                    "refusal_f1",
                    "latency_p50_ms",
                    "latency_p95_ms",
                    "n",
                    "n_answerable",
                    "n_unanswerable",
                )
            }
            for aid, m in by_arm.items()
        },
        "by_category": by_category,
        "latency_p50_ms": (sum(lat_p50) / len(lat_p50)) if lat_p50 else 0.0,
        "latency_p95_ms": max(lat_p95) if lat_p95 else 0.0,
        "peak_rss_mb": _rss_mb(),
        "index_bytes": _index_bytes_estimate(),
        "selection": {
            "winner_id": selection.get("winner_id")
            or (hc.get("winner_id"))
            or selection.get("primary", {}).get("arm_id"),
            "rule_hash": selection.get("selection_rules_sha256") or rules_content_hash(),
            "holdout_passed": bool(hc.get("pass")) if hc else None,
            "baseline_arm_id": selection.get("baseline_arm_id") or hc.get("baseline_arm_id"),
            "deltas": deltas,
        },
        "regression": regression_metrics,
        "holdout_confirmation_pass": hc.get("pass") if hc else None,
    }
    path = RESULTS_DIR / "quality_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
