"""Evaluate labeled set under a config; produce metrics records."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rag_bench.config_load import LABELS_PATH, RESULTS_DIR, ensure_dirs
from rag_bench.graph import run_pipeline
from rag_bench.index import RAGIndex
from rag_bench.metrics import attribution_hit, evaluate_run, recall_at_k

# Keys omitted from tracked JSON so full recompute stays path/time-agnostic
# and bit-stable across CPU/OS/Python float noise.
_VOLATILE_METRIC_KEYS = frozenset(
    {
        "latencies_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_ms",
        "top1_scores",  # embedding float noise across platforms
    }
)


def stable_metrics_for_disk(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop wall-clock latency fields from metrics destined for tracked files."""
    return {k: v for k, v in metrics.items() if k not in _VOLATILE_METRIC_KEYS}


def load_labels(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or LABELS_PATH
    # Holdout seal: route through holdout module when path is sealed holdout
    from rag_bench.holdout import HOLDOUT_LABELS, load_labels as holdout_load

    try:
        if p is not None and Path(p).resolve() == HOLDOUT_LABELS.resolve():
            return holdout_load(p)
    except Exception:
        pass
    # also match by suffix path
    if p is not None and "holdout" in Path(p).parts and Path(p).name == "labels.jsonl":
        return holdout_load(p)

    labels = []
    with Path(p).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(json.loads(line))
    return labels


def evaluate_config(
    cfg: dict[str, Any],
    *,
    labels: list[dict[str, Any]] | None = None,
    index: RAGIndex | None = None,
    collect_latencies: bool = True,
) -> dict[str, Any]:
    """Run full RAG pipeline on all labels; return metrics + per-qid details + hit vectors."""
    labels = labels if labels is not None else load_labels()
    strategy = str(cfg.get("chunk_strategy", "fixed_512"))
    embeddings = str(cfg.get("embeddings", "hash"))
    if index is None:
        index = RAGIndex.build(strategy_name=strategy, embeddings_kind=embeddings)

    records = []
    details = []
    hit_vector: list[int] = []
    attr_vector: list[int] = []
    qids: list[str] = []
    latencies_ms: list[float] = []
    top1_scores: list[float] = []

    for lab in labels:
        query = lab["question"]
        t0 = time.perf_counter()
        result = run_pipeline(
            query,
            index,
            top_k=int(cfg.get("top_k", 4)),
            threshold=cfg.get("threshold"),
            retriever=str(cfg.get("retriever", "dense")),
            retrieval_enabled=bool(cfg.get("retrieval", True)),
            rerank_enabled=bool(cfg.get("rerank", True)),
            rrf_k=int(cfg.get("rrf_k", 60)),
            query_strategy=str(cfg.get("query_strategy", "raw")),
            abstain=str(cfg.get("abstain", "none")),
            median_dev=cfg.get("median_dev"),
            margin=float(cfg.get("margin", 0.05)),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if collect_latencies:
            latencies_ms.append(elapsed_ms)

        retrieved_ids = [
            d.get("chunk_id")
            for d in (result.get("retrieved_docs") or [])
            if d.get("chunk_id")
        ]
        # Prefer post-rerank order for recall if present
        reranked = result.get("reranked_docs") or []
        if reranked:
            retrieved_ids = [d.get("chunk_id") for d in reranked if d.get("chunk_id")]

        answerable = lab.get("answerable")
        if answerable is None:
            answerable = bool(lab.get("gold_spans") or lab.get("must_contain") or lab.get("gold_doc_ids"))

        rec = {
            "retrieved_chunk_ids": retrieved_ids,
            "source_chunk_ids": result.get("source_chunk_ids") or [],
            "answer": result.get("answer") or "",
            "gold_spans": lab.get("gold_spans") or [],
            "must_contain": lab.get("must_contain") or [],
            "gold_doc_ids": lab.get("gold_doc_ids") or [],
            "answerable": bool(answerable),
            "stage": result.get("stage"),
            "category": lab.get("category"),
        }
        records.append(rec)
        top1_scores.append(float(result.get("top1_score") or 0.0))

        if answerable:
            hit = int(
                recall_at_k(
                    retrieved_ids,
                    lab.get("gold_spans") or [],
                    lab.get("gold_doc_ids"),
                )
            )
        else:
            hit = 0
        hit_vector.append(hit)
        qid = str(lab.get("qid") or "")
        qids.append(qid)
        details.append(
            {
                "qid": qid,
                "category": lab.get("category"),
                "answerable": bool(answerable),
                "retrieved_chunk_ids": retrieved_ids,
                "source_chunk_ids": rec["source_chunk_ids"],
                "answer": rec["answer"],
                "recall_hit": hit,
                "latency_ms": elapsed_ms,
                "top1_score": result.get("top1_score"),
                "top2_score": result.get("top2_score"),
                "stage": result.get("stage"),
            }
        )

    metrics = evaluate_run(records)
    for rec in records:
        if not rec.get("answerable", True):
            attr_vector.append(0)
            continue
        attr_vector.append(
            int(
                attribution_hit(
                    rec.get("source_chunk_ids") or [],
                    rec.get("answer") or "",
                    rec.get("gold_spans") or [],
                    rec.get("must_contain") or [],
                )
            )
        )
    metrics["config"] = {
        k: cfg.get(k)
        for k in (
            "chunk_strategy",
            "top_k",
            "rerank",
            "retrieval",
            "threshold",
            "embeddings",
            "retriever",
            "query_strategy",
            "abstain",
            "rrf_k",
            "arm_id",
        )
    }
    metrics["hit_vector"] = hit_vector
    metrics["attr_vector"] = attr_vector
    metrics["qids"] = qids
    metrics["top1_scores"] = top1_scores
    if latencies_ms:
        sorted_lat = sorted(latencies_ms)
        metrics["latency_p50_ms"] = sorted_lat[len(sorted_lat) // 2]
        metrics["latency_p95_ms"] = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
        metrics["latencies_ms"] = latencies_ms
    else:
        metrics["latency_p50_ms"] = 0.0
        metrics["latency_p95_ms"] = 0.0

    # by_category
    by_cat: dict[str, dict[str, Any]] = {}
    for lab, rec, hit in zip(labels, records, hit_vector):
        cat = str(lab.get("category") or "unknown")
        slot = by_cat.setdefault(cat, {"n": 0, "recall_hits": 0, "answerable": 0})
        slot["n"] += 1
        if rec.get("answerable"):
            slot["answerable"] += 1
            slot["recall_hits"] += hit
    for cat, slot in by_cat.items():
        slot["recall_at_k"] = (
            slot["recall_hits"] / slot["answerable"] if slot["answerable"] else 0.0
        )
    metrics["by_category"] = by_cat

    return {
        "metrics": metrics,
        "details": details,
        "records": records,
        "hit_vector": hit_vector,
        "attr_vector": attr_vector,
        "qids": qids,
        "top1_scores": top1_scores,
    }


def write_metrics_json(metrics: dict[str, Any], path: Path | None = None) -> Path:
    ensure_dirs()
    out = path or (RESULTS_DIR / "metrics.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out
