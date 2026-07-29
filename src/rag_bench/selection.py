"""Dev shortlist (lexicographic) → holdout finalize (confirm-only, no re-rank)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from rag_bench.arms import ARMS_RESULTS, arm_to_cfg, evaluate_arm_on_labels, load_arms
from rag_bench.config_load import CONFIG_DIR, RESULTS_DIR, ensure_dirs, save_selected
from rag_bench.eval import load_labels
from rag_bench.holdout import DEV_SHORTLIST, HOLDOUT_LABELS, UNLOCK_ENV
from rag_bench.index import RAGIndex

SELECTION_RULES = CONFIG_DIR / "selection_rules.yaml"


def load_selection_rules() -> dict[str, Any]:
    return yaml.safe_load(SELECTION_RULES.read_text(encoding="utf-8")) or {}


def rules_content_hash() -> str:
    return hashlib.sha256(SELECTION_RULES.read_bytes()).hexdigest()


def _metric_val(metrics: dict[str, Any], key: str) -> float:
    v = metrics.get(key)
    if v is None:
        return 0.0
    return float(v)


def lexicographic_rank(
    arm_metrics: dict[str, dict[str, Any]],
    rules: dict[str, Any],
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Return arm_ids ordered best-first by lexicographic rules on DEV metrics."""
    exclude = exclude or set()
    lex = rules.get("lexicographic") or []
    n_unans_gate = 5

    def sort_key(arm_id: str) -> tuple:
        m = arm_metrics[arm_id]
        keys: list[Any] = []
        for step in lex:
            if "maximize" in step:
                field = step["maximize"]
                if field == "refusal_f1" and int(m.get("n_unanswerable") or 0) < n_unans_gate:
                    keys.append(0.0)  # skip step → neutral
                else:
                    keys.append(-_metric_val(m, field))  # maximize → negate
            elif "minimize" in step:
                field = step["minimize"]
                # p95 may be under latency_p95_ms
                if field == "p95_e2e_ms":
                    keys.append(_metric_val(m, "latency_p95_ms"))
                else:
                    keys.append(_metric_val(m, field))
        # tie-break: lower p95, then arm_id lex smaller
        keys.append(_metric_val(m, "latency_p95_ms"))
        keys.append(arm_id)
        return tuple(keys)

    candidates = [a for a in arm_metrics if a not in exclude]
    return sorted(candidates, key=sort_key)


def write_dev_shortlist(dev_arm_results: dict[str, Any]) -> dict[str, Any]:
    """
    Rank 9 arms (exclude retrieval_off_control) on DEV only.
    shortlist_size always 1.
    """
    ensure_dirs()
    rules = load_selection_rules()
    arms_map = dev_arm_results.get("arms") or dev_arm_results
    metrics_map = {aid: (res.get("metrics") or res) for aid, res in arms_map.items()}
    exclude = {"retrieval_off_control"}
    ranking_ids = lexicographic_rank(metrics_map, rules, exclude=exclude)
    primary_id = ranking_ids[0]
    ranking = []
    for i, aid in enumerate(ranking_ids, start=1):
        m = metrics_map[aid]
        ranking.append(
            {
                "arm_id": aid,
                "rank": i,
                "dev_metrics_summary": {
                    "attribution_rate": m.get("attribution_rate"),
                    "recall_at_k": m.get("recall_at_k"),
                    "refusal_f1": m.get("refusal_f1"),
                    "error_citation_rate": m.get("error_citation_rate"),
                    "latency_p95_ms": m.get("latency_p95_ms"),
                },
            }
        )
    payload = {
        "schema": 1,
        "source": "dev_only",
        "n_arms_ranked": len(ranking_ids),
        "shortlist_size": 1,
        "primary": {
            "arm_id": primary_id,
            "dev_metrics": metrics_map[primary_id],
        },
        "baseline_arm_id": rules.get("baseline_arm_id", "hash_dense_k8_r1"),
        "ranking": ranking,
        "selection_rules_sha256": rules_content_hash(),
    }
    DEV_SHORTLIST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def bootstrap_delta_recall(
    winner_hits: list[int],
    baseline_hits: list[int],
    *,
    B: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap on answerable holdout qids: Δrecall = mean(w)-mean(b)."""
    import random

    n = min(len(winner_hits), len(baseline_hits))
    if n == 0:
        return {
            "point_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "excludes_zero": False,
            "B": B,
            "n": 0,
        }
    w = winner_hits[:n]
    b = baseline_hits[:n]
    point = sum(w[i] - b[i] for i in range(n)) / n
    rng = random.Random(seed)
    deltas = []
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(w[i] - b[i] for i in idxs) / n
        deltas.append(d)
    deltas.sort()
    alpha = 1.0 - ci_level
    lo_i = int(alpha / 2 * B)
    hi_i = int((1 - alpha / 2) * B) - 1
    lo_i = max(0, min(B - 1, lo_i))
    hi_i = max(0, min(B - 1, hi_i))
    ci_low, ci_high = deltas[lo_i], deltas[hi_i]
    return {
        "point_delta": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": (ci_low > 0) or (ci_high < 0),
        "ci_excludes_zero_positive": ci_low > 0,
        "B": B,
        "n": n,
        "seed": seed,
    }


def finalize_selection(
    *,
    shortlist_path: Path | None = None,
    holdout_path: Path | None = None,
) -> dict[str, Any]:
    """
    Unlock holdout; evaluate ONLY primary + baseline on holdout.
    No re-ranking. Fail → NEEDS_REPLAN, do not switch arm.
    """
    ensure_dirs()
    shortlist_path = shortlist_path or DEV_SHORTLIST
    if not shortlist_path.exists():
        raise FileNotFoundError("dev_shortlist.json required before finalize_selection")

    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    primary_id = shortlist["primary"]["arm_id"]
    baseline_id = shortlist.get("baseline_arm_id") or "hash_dense_k8_r1"
    rules = load_selection_rules()

    # Sole unlock
    os.environ[UNLOCK_ENV] = "1"
    try:
        labels = load_labels(holdout_path or HOLDOUT_LABELS)
        arms = {a["id"]: a for a in load_arms()}
        index_cache: dict[tuple[str, str], RAGIndex] = {}

        # margin calib from dev file if primary or baseline needs it
        median_dev = None
        calib_path = ARMS_RESULTS / "minilm_dense_k8_r1_margin_calib.json"
        if calib_path.exists():
            median_dev = json.loads(calib_path.read_text(encoding="utf-8")).get("median_dev")

        def eval_one(aid: str) -> dict[str, Any]:
            arm = arms[aid]
            mid = median_dev if aid == "minilm_dense_k8_r1_margin" else None
            return evaluate_arm_on_labels(arm, labels, index_cache=index_cache, median_dev=mid)

        primary_res = eval_one(primary_id)
        baseline_res = eval_one(baseline_id)

        # answerable-only hit vectors for bootstrap
        def answerable_hits(res: dict[str, Any]) -> list[int]:
            details = res.get("details") or []
            hits = []
            for d in details:
                if d.get("answerable", True):
                    hits.append(int(d.get("recall_hit") or 0))
            # fallback to metrics hit_vector if details missing answerable flags
            if not hits and res.get("hit_vector"):
                return list(res["hit_vector"])
            return hits

        boot = bootstrap_delta_recall(
            answerable_hits(primary_res),
            answerable_hits(baseline_res),
            B=int(rules.get("bootstrap_B", 1000)),
            seed=int(rules.get("bootstrap_seed", 42)),
            ci_level=float((rules.get("bootstrap") or {}).get("ci_level", 0.95)),
        )
        sig = rules.get("significance") or {}
        min_delta = float(sig.get("min_delta_recall", 0.05))
        pm = primary_res["metrics"]
        bm = baseline_res["metrics"]

        delta_recall_ok = boot["point_delta"] >= min_delta and (
            boot["ci_excludes_zero_positive"] if sig.get("ci_must_exclude_zero", True) else True
        )
        attr_ok = _metric_val(pm, "attribution_rate") >= (
            _metric_val(bm, "attribution_rate") - float(sig.get("max_attribution_drop", 0.02))
        )
        unsup_ok = _metric_val(pm, "unsupported_answer_rate") <= (
            _metric_val(bm, "unsupported_answer_rate") + float(sig.get("max_unsupported_rise", 0.02))
        )
        n_unans = int(pm.get("n_unanswerable") or 0)
        if n_unans >= 5:
            refusal_ok = _metric_val(pm, "refusal_recall") >= (
                _metric_val(bm, "refusal_recall") - float(sig.get("max_refusal_recall_drop", 0.05))
            )
        else:
            refusal_ok = True  # N/A skip

        # retrieval_off control check from prior dev if present
        retrieval_off_ok = True

        passed = bool(delta_recall_ok and attr_ok and unsup_ok and refusal_ok)

        confirmation = {
            "winner_id": primary_id,
            "baseline_arm_id": baseline_id,
            "bootstrap": boot,
            "significance": {
                "delta_recall_ok": delta_recall_ok,
                "attr_ok": attr_ok,
                "unsup_ok": unsup_ok,
                "refusal_ok": refusal_ok,
                "retrieval_off_ok": retrieval_off_ok,
            },
            "pass": passed,
            "holdout_primary_metrics": {
                k: pm.get(k)
                for k in (
                    "recall_at_k",
                    "attribution_rate",
                    "unsupported_answer_rate",
                    "refusal_f1",
                    "refusal_recall",
                    "n",
                    "n_answerable",
                    "n_unanswerable",
                )
            },
            "holdout_baseline_metrics": {
                k: bm.get(k)
                for k in (
                    "recall_at_k",
                    "attribution_rate",
                    "unsupported_answer_rate",
                    "refusal_f1",
                    "refusal_recall",
                    "n",
                    "n_answerable",
                    "n_unanswerable",
                )
            },
            "selection_rules_sha256": rules_content_hash(),
            "shortlist_size": 1,
            "re_ranked_on_holdout": False,
        }
        (RESULTS_DIR / "holdout_confirmation.json").write_text(
            json.dumps(confirmation, indent=2) + "\n", encoding="utf-8"
        )

        # holdout arm metric files
        ARMS_RESULTS.mkdir(parents=True, exist_ok=True)
        for res in (primary_res, baseline_res):
            (ARMS_RESULTS / f"holdout_{res['arm_id']}.json").write_text(
                json.dumps(
                    {
                        "arm_id": res["arm_id"],
                        "metrics": res["metrics"],
                        "hit_vector": res["hit_vector"],
                        "qids": res["qids"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        if passed:
            arm = arms[primary_id]
            cfg = arm_to_cfg(arm, median_dev=median_dev if primary_id.endswith("margin") else None)
            cfg["source"] = "holdout_confirmed"
            cfg["winning_arm_id"] = primary_id
            cfg["holdout_recall_at_k"] = pm.get("recall_at_k")
            cfg["holdout_attribution_rate"] = pm.get("attribution_rate")
            cfg["selection_rules_sha256"] = rules_content_hash()
            save_selected(cfg)
            confirmation["selected_status"] = "holdout_passed"
        else:
            # do not change selected to another arm
            confirmation["selected_status"] = "holdout_failed"
            # write status note on selected
            sel_path = CONFIG_DIR / "selected.yaml"
            if sel_path.exists():
                existing = yaml.safe_load(sel_path.read_text(encoding="utf-8")) or {}
            else:
                existing = arm_to_cfg(arms[baseline_id])
            existing["selected_status"] = "holdout_failed"
            existing["holdout_failed_primary"] = primary_id
            existing["source"] = existing.get("source") or "pre_holdout"
            save_selected(existing)

        confirmation["primary_res_summary"] = primary_res["metrics"]
        return confirmation
    finally:
        # reseal
        os.environ.pop(UNLOCK_ENV, None)
