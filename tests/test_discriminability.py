"""Unit tests for SE / Wilson CI / paired deltas."""

from __future__ import annotations

from rag_bench.discriminability import (
    cell_uncertainty,
    paired_delta,
    proportion_se,
    wilson_ci,
    analyze_rows,
)


def test_proportion_se_known():
    # p=0.5, n=100 → se=0.05
    se = proportion_se(0.5, 100)
    assert abs(se - 0.05) < 1e-9


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0.97, 35)
    assert 0.0 <= lo < 0.97 < hi <= 1.0


def test_cell_uncertainty_from_vector():
    hv = [1] * 30 + [0] * 5
    u = cell_uncertainty(0, hit_vector=hv)
    assert u["n"] == 35
    assert abs(u["p_hat"] - 30 / 35) < 1e-12
    assert u["wilson_95"][0] < u["p_hat"] < u["wilson_95"][1]


def test_paired_delta_discriminative():
    a = [1, 1, 1, 1, 0, 0, 1, 1]
    b = [0, 0, 0, 0, 0, 0, 0, 0]
    d = paired_delta(a, b)
    assert d["mean_delta"] > 0
    assert d["discriminative"] is True


def test_paired_delta_nondiscriminative():
    a = [1, 0, 1, 0, 1, 0]
    b = [1, 0, 1, 0, 1, 0]
    d = paired_delta(a, b)
    assert d["mean_delta"] == 0.0
    assert d["discriminative"] is False


def test_analyze_rows_retrieval_discriminative():
    rows = []
    for i, ret in enumerate([True, True, False, False]):
        hv = [1] * 20 if ret else [0] * 20
        rows.append(
            {
                "cell_id": i,
                "retrieval": ret,
                "rerank": True,
                "chunk_strategy": "fixed_256",
                "top_k": 4,
                "threshold": "",
                "embeddings": "hash",
                "recall_at_k": sum(hv) / len(hv),
                "n": len(hv),
                "hit_vector": hv,
            }
        )
    # make rerank non-disc
    rows[0]["rerank"] = True
    rows[1]["rerank"] = False
    rows[1]["hit_vector"] = list(rows[0]["hit_vector"])
    rows[1]["recall_at_k"] = rows[0]["recall_at_k"]
    analysis = analyze_rows(rows)
    assert analysis["factors"]["retrieval"]["discriminative"] is True
