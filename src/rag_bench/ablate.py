"""Run ablation grid and select winning config from measured metrics."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from rag_bench.config_load import (
    RESULTS_DIR,
    ensure_dirs,
    load_ablation_grid,
    save_selected,
)
from rag_bench.eval import evaluate_config, load_labels
from rag_bench.index import RAGIndex


def _grid_cells(grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = ["retrieval", "rerank", "chunk_strategy", "top_k", "threshold"]
    values = []
    for k in keys:
        if k not in grid:
            raise KeyError(f"ablation grid missing key: {k}")
        values.append(list(grid[k]))
    # embeddings may be scalar or list
    emb_levels = grid.get("embeddings", "hash")
    if not isinstance(emb_levels, list):
        emb_levels = [emb_levels]
    retriever = grid.get("retriever", "dense")
    cells = []
    for combo in itertools.product(*values):
        base = dict(zip(keys, combo))
        for emb in emb_levels:
            cell = dict(base)
            cell["embeddings"] = emb
            cell["retriever"] = retriever
            cells.append(cell)
    return cells


def run_ablations(
    grid: dict[str, Any] | None = None,
    *,
    labels: list[dict[str, Any]] | None = None,
    results_dir: Path | None = None,
    save_selected_yaml: bool = True,
    store_hit_vectors: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    grid = grid or load_ablation_grid()
    labels = labels if labels is not None else load_labels()
    cells = _grid_cells(grid)
    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache indexes per (chunk strategy, embeddings)
    index_cache: dict[tuple[str, str], RAGIndex] = {}
    rows: list[dict[str, Any]] = []

    for i, cell in enumerate(cells):
        strategy = str(cell["chunk_strategy"])
        emb = str(cell.get("embeddings", "hash"))
        key = (strategy, emb)
        if key not in index_cache:
            index_cache[key] = RAGIndex.build(
                strategy_name=strategy,
                embeddings_kind=emb,
            )
        result = evaluate_config(cell, labels=labels, index=index_cache[key])
        m = result["metrics"]
        row = {
            "cell_id": i,
            "retrieval": cell["retrieval"],
            "rerank": cell["rerank"],
            "chunk_strategy": cell["chunk_strategy"],
            "top_k": cell["top_k"],
            "threshold": cell["threshold"] if cell["threshold"] is not None else "",
            "embeddings": cell.get("embeddings", "hash"),
            "retriever": cell.get("retriever", "dense"),
            "recall_at_k": m["recall_at_k"],
            "attribution_rate": m["attribution_rate"],
            "n": m["n"],
        }
        if store_hit_vectors:
            row["hit_vector"] = result.get("hit_vector") or m.get("hit_vector")
            row["attr_vector"] = result.get("attr_vector") or m.get("attr_vector")
            row["qids"] = result.get("qids") or m.get("qids")
        rows.append(row)
        print(
            f"ablation cell {i+1}/{len(cells)}: "
            f"ret={cell['retrieval']} rerank={cell['rerank']} "
            f"chunk={cell['chunk_strategy']} k={cell['top_k']} thr={cell['threshold']} "
            f"emb={emb} "
            f"→ recall={m['recall_at_k']:.3f} attr={m['attribution_rate']:.3f}"
        )

    # Write CSV (omit large vectors from CSV for readability; keep in jsonl sidecar)
    csv_path = out_dir / "ablation_table.csv"
    csv_fields = [
        "cell_id",
        "retrieval",
        "rerank",
        "chunk_strategy",
        "top_k",
        "threshold",
        "embeddings",
        "retriever",
        "recall_at_k",
        "attribution_rate",
        "n",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Full rows with hit vectors
    hits_path = out_dir / "ablation_hits.json"
    with hits_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "cell_id": r["cell_id"],
                    "hit_vector": r.get("hit_vector"),
                    "attr_vector": r.get("attr_vector"),
                    "qids": r.get("qids"),
                    "recall_at_k": r["recall_at_k"],
                    "config": {
                        k: r[k]
                        for k in (
                            "retrieval",
                            "rerank",
                            "chunk_strategy",
                            "top_k",
                            "threshold",
                            "embeddings",
                            "retriever",
                        )
                    },
                }
                for r in rows
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")

    winner = select_winner(rows)
    summary = {
        "n_cells": len(rows),
        "winner": winner,
        "selection_rule": (
            "Among retrieval=true: max recall_at_k, then attribution_rate; "
            "prefer rerank true / simpler threshold on ties when non-discriminative"
        ),
        "csv": str(csv_path),
        "hits": str(hits_path),
    }
    summary_path = out_dir / "ablation_summary.json"
    # winner may contain lists — strip for JSON summary
    win_serializable = {k: v for k, v in winner.items() if k not in ("hit_vector", "attr_vector", "qids")}
    summary["winner"] = win_serializable
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    selected = {
        "chunk_strategy": winner["chunk_strategy"],
        "top_k": int(winner["top_k"]),
        "rerank": _as_bool(winner["rerank"]),
        "retrieval": _as_bool(winner["retrieval"]),
        "threshold": None if winner["threshold"] in ("", None) else float(winner["threshold"]),
        "embeddings": winner.get("embeddings", "hash"),
        "retriever": winner.get("retriever", "dense"),
        "source": "ablation_table.csv",
        "winning_cell_id": winner["cell_id"],
        "winning_recall_at_k": winner["recall_at_k"],
        "winning_attribution_rate": winner["attribution_rate"],
    }
    if save_selected_yaml:
        save_selected(selected)

    return {"rows": rows, "summary": summary, "selected": selected, "results_dir": str(out_dir)}


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Among retrieval=true: max recall@k, then attribution; prefer simpler ties."""
    if not rows:
        raise ValueError("no ablation rows")
    candidates = [r for r in rows if _as_bool(r.get("retrieval", True))]
    if not candidates:
        candidates = rows

    def key(r: dict[str, Any]):
        thr = r["threshold"]
        thr_simple = 1 if thr in ("", None) else 0  # prefer null threshold on metric ties
        rerank_pref = 0  # prefer rerank=false on pure metric ties (simpler) — metrics first
        return (
            float(r["recall_at_k"]),
            float(r["attribution_rate"]),
            thr_simple,
            1 if _as_bool(r["rerank"]) else 0,
            int(r["top_k"]),
        )

    return max(candidates, key=key)


def factor_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate mean metrics per factor value for tradeoffs.md.

    For factors other than `retrieval`, means are computed only on retrieval=True
    cells so zeros from the retrieval-off baseline do not dilute comparisons.
    """
    factors = ["retrieval", "rerank", "chunk_strategy", "top_k", "threshold"]
    if any("embeddings" in r for r in rows):
        factors.append("embeddings")
    out: dict[str, Any] = {}
    for fac in factors:
        if not any(fac in r for r in rows):
            continue
        if fac == "retrieval":
            subset = rows
        else:
            subset = [r for r in rows if _as_bool(r.get("retrieval", True))]
        buckets: dict[str, list[dict]] = {}
        for r in subset:
            if fac not in r:
                continue
            val = r[fac]
            key = "" if val is None else str(val)
            buckets.setdefault(key, []).append(r)
        stats = {}
        for k, rs in buckets.items():
            stats[k] = {
                "n": len(rs),
                "mean_recall": sum(float(x["recall_at_k"]) for x in rs) / len(rs),
                "mean_attribution": sum(float(x["attribution_rate"]) for x in rs) / len(rs),
                "best_cell_id": max(
                    rs,
                    key=lambda x: (float(x["recall_at_k"]), float(x["attribution_rate"])),
                )["cell_id"],
            }
        out[fac] = stats
    return out
