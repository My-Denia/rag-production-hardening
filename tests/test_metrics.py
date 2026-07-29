"""Metrics unit tests with fixtures (implementation-independent judges)."""

from __future__ import annotations

from rag_bench.metrics import (
    attribution_hit,
    evaluate_run,
    recall_at_k,
)


def test_recall_hit_on_overlapping_span():
    gold = [{"doc_id": "docA", "start": 0, "end": 50}]
    retrieved = ["docA::span:10:40", "docB::span:0:20"]
    assert recall_at_k(retrieved, gold) == 1.0


def test_recall_miss():
    gold = [{"doc_id": "docA", "start": 0, "end": 50}]
    retrieved = ["docB::span:0:20"]
    assert recall_at_k(retrieved, gold) == 0.0


def test_attribution_requires_gold_cite_and_must_contain():
    gold = [{"doc_id": "docA", "start": 0, "end": 100}]
    sources = ["docA::span:5:50"]
    answer = "The limit is 20 days."
    assert attribution_hit(sources, answer, gold, ["20"]) == 1.0
    assert attribution_hit(sources, answer, gold, ["99"]) == 0.0
    assert attribution_hit(["docB::span:0:10"], answer, gold, ["20"]) == 0.0


def test_evaluate_run_aggregates():
    records = [
        {
            "retrieved_chunk_ids": ["d::span:0:10"],
            "source_chunk_ids": ["d::span:0:10"],
            "answer": "value is 5",
            "gold_spans": [{"doc_id": "d", "start": 0, "end": 5}],
            "must_contain": ["5"],
            "gold_doc_ids": ["d"],
        },
        {
            "retrieved_chunk_ids": ["x::span:0:10"],
            "source_chunk_ids": ["x::span:0:10"],
            "answer": "nope",
            "gold_spans": [{"doc_id": "d", "start": 0, "end": 5}],
            "must_contain": ["5"],
            "gold_doc_ids": ["d"],
        },
    ]
    m = evaluate_run(records)
    assert m["n"] == 2
    assert m["recall_at_k"] == 0.5
    assert m["attribution_rate"] == 0.5
    assert isinstance(m["recall_at_k"], float)
    assert isinstance(m["attribution_rate"], float)
