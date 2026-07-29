"""Frozen arm evaluation on DEV only; margin calibration; determinism check."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from rag_bench.config_load import CONFIG_DIR, RESULTS_DIR, ensure_dirs
from rag_bench.eval import evaluate_config, load_labels
from rag_bench.holdout import DEV_LABELS
from rag_bench.index import RAGIndex

ARMS_PATH = CONFIG_DIR / "arms.yaml"
ARMS_RESULTS = RESULTS_DIR / "arms"


def load_arms() -> list[dict[str, Any]]:
    data = yaml.safe_load(ARMS_PATH.read_text(encoding="utf-8")) or {}
    arms = data.get("arms") or []
    if len(arms) != 10:
        raise ValueError(f"arms.yaml must have exactly 10 arms, found {len(arms)}")
    return arms


def arm_to_cfg(arm: dict[str, Any], *, median_dev: float | None = None) -> dict[str, Any]:
    cfg = {
        "arm_id": arm["id"],
        "embeddings": arm.get("embeddings", "hash"),
        "retriever": arm.get("retriever", "dense"),
        "top_k": int(arm.get("top_k", 8)),
        "rerank": bool(arm.get("rerank", False)),
        "chunk_strategy": arm.get("chunk_strategy", "fixed_512"),
        "query_strategy": arm.get("query_strategy", "raw"),
        "abstain": arm.get("abstain", "none"),
        "retrieval": bool(arm.get("retrieval", True)),
        "threshold": arm.get("threshold"),
        "rrf_k": int(arm.get("rrf_k", 60)),
        "margin": 0.05,
    }
    if median_dev is not None:
        cfg["median_dev"] = median_dev
    return cfg


def _index_cache_key(arm: dict[str, Any]) -> tuple[str, str]:
    return (str(arm.get("chunk_strategy", "fixed_512")), str(arm.get("embeddings", "hash")))


def evaluate_arm_on_labels(
    arm: dict[str, Any],
    labels: list[dict[str, Any]],
    *,
    index_cache: dict[tuple[str, str], RAGIndex] | None = None,
    median_dev: float | None = None,
) -> dict[str, Any]:
    cache = index_cache if index_cache is not None else {}
    key = _index_cache_key(arm)
    if key not in cache:
        # retrieval_off still needs an index object but empty retrieval
        cache[key] = RAGIndex.build(strategy_name=key[0], embeddings_kind=key[1])
    cfg = arm_to_cfg(arm, median_dev=median_dev)
    ev = evaluate_config(cfg, labels=labels, index=cache[key])
    m = ev["metrics"]
    m["arm_id"] = arm["id"]
    return {
        "arm_id": arm["id"],
        "cfg": cfg,
        "metrics": m,
        "hit_vector": ev["hit_vector"],
        "attr_vector": ev["attr_vector"],
        "qids": ev["qids"],
        "details": ev["details"],
        "top1_scores": ev.get("top1_scores") or [],
    }


def calibrate_margin_median(arm: dict[str, Any], labels: list[dict[str, Any]], index: RAGIndex) -> float:
    """Collect top1 scores after retrieve+rerank on dev; return median."""
    cfg = arm_to_cfg(arm)
    # force no abstain during calibration
    cfg["abstain"] = "none"
    cfg.pop("median_dev", None)
    ev = evaluate_config(cfg, labels=labels, index=index)
    scores = [float(s) for s in (ev.get("top1_scores") or [])]
    if not scores:
        return 0.0
    return float(statistics.median(scores))


def run_all_arms_dev(
    *,
    labels_path: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    ARMS_RESULTS.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path or DEV_LABELS)
    arms = load_arms()
    index_cache: dict[tuple[str, str], RAGIndex] = {}
    results: dict[str, Any] = {}

    # Pre-calibrate margin arm before its eval
    margin_arm = next((a for a in arms if a["id"] == "minilm_dense_k8_r1_margin"), None)
    median_dev = None
    if margin_arm is not None:
        key = _index_cache_key(margin_arm)
        if key not in index_cache:
            index_cache[key] = RAGIndex.build(strategy_name=key[0], embeddings_kind=key[1])
        median_dev = calibrate_margin_median(margin_arm, labels, index_cache[key])
        calib = {
            "arm_id": "minilm_dense_k8_r1_margin",
            "median_dev": median_dev,
            "n_dev": len(labels),
            "margin": 0.05,
        }
        if write:
            (ARMS_RESULTS / "minilm_dense_k8_r1_margin_calib.json").write_text(
                json.dumps(calib, indent=2) + "\n", encoding="utf-8"
            )

    for arm in arms:
        mid = median_dev if arm["id"] == "minilm_dense_k8_r1_margin" else None
        print(f"  arm eval: {arm['id']} ...", flush=True)
        res = evaluate_arm_on_labels(arm, labels, index_cache=index_cache, median_dev=mid)
        results[arm["id"]] = res
        if write:
            out = {
                "arm_id": res["arm_id"],
                "cfg": res["cfg"],
                "metrics": res["metrics"],
                "hit_vector": res["hit_vector"],
                "attr_vector": res["attr_vector"],
                "qids": res["qids"],
                "n": len(labels),
            }
            (ARMS_RESULTS / f"dev_{arm['id']}.json").write_text(
                json.dumps(out, indent=2) + "\n", encoding="utf-8"
            )

    return {"arms": results, "median_dev": median_dev, "n_dev": len(labels)}


def determinism_check(
    arm_id: str = "minilm_dense_k8_r1",
    *,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    labels = load_labels(labels_path or DEV_LABELS)
    arms = {a["id"]: a for a in load_arms()}
    arm = arms[arm_id]
    key = _index_cache_key(arm)
    index = RAGIndex.build(strategy_name=key[0], embeddings_kind=key[1])
    r1 = evaluate_arm_on_labels(arm, labels, index_cache={key: index})
    r2 = evaluate_arm_on_labels(arm, labels, index_cache={key: index})
    h1 = hashlib.sha256(json.dumps(r1["hit_vector"]).encode()).hexdigest()
    h2 = hashlib.sha256(json.dumps(r2["hit_vector"]).encode()).hexdigest()
    payload = {
        "ok": h1 == h2,
        "arm_id": arm_id,
        "hash_run1": h1,
        "hash_run2": h2,
        "identical": h1 == h2,
    }
    ensure_dirs()
    (RESULTS_DIR / "determinism_check.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def hybrid_differs_from_dense(dev_results: dict[str, Any]) -> dict[str, Any]:
    """Check hybrid ≠ dense on ≥1 qid."""
    arms = dev_results.get("arms") or dev_results
    dense = arms.get("minilm_dense_k8_r1") or {}
    hybrid = arms.get("hybrid_rrf_k8_r1") or {}
    hv_d = dense.get("hit_vector") or []
    hv_h = hybrid.get("hit_vector") or []
    qids = dense.get("qids") or hybrid.get("qids") or []
    diffs = []
    for i, (a, b) in enumerate(zip(hv_d, hv_h)):
        if a != b:
            diffs.append(qids[i] if i < len(qids) else str(i))
    return {
        "differs": len(diffs) > 0 or (hv_d != hv_h and bool(hv_d or hv_h)),
        "n_diff_qids": len(diffs),
        "sample_qids": diffs[:5],
        # also compare retrieved sets via details if hit vectors equal
        "dense_n": len(hv_d),
        "hybrid_n": len(hv_h),
    }
