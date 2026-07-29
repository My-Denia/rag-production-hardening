"""Retrieval unit tests — gold doc appears for an easy query."""

from __future__ import annotations

from rag_bench.chunking import chunk_documents, gold_span_to_chunk_ids, load_corpus
from rag_bench.index import RAGIndex
from rag_bench.retrieve import RetrieveConfig, retrieve


def test_chunk_strategies_produce_span_ids():
    for name in ("fixed_256", "fixed_512", "recursive_400_80"):
        chunks = chunk_documents(strategy_name=name)
        assert len(chunks) >= 8
        for ch in chunks:
            assert "chunk_id" in ch.metadata
            assert "::span:" in ch.metadata["chunk_id"]
            assert ch.metadata["start"] < ch.metadata["end"]


def test_easy_query_retrieves_gold_doc():
    index = RAGIndex.build(strategy_name="fixed_512", embeddings_kind="hash")
    docs = retrieve(
        "How many annual paid leave days for full-time employees with less than five years?",
        index,
        RetrieveConfig(top_k=4, retrieval_enabled=True, retriever="dense"),
    )
    assert len(docs) >= 1
    doc_ids = {d.metadata.get("doc_id") for d in docs}
    assert "hr_leave_policy" in doc_ids


def test_gold_span_maps_to_chunks():
    chunks = chunk_documents(strategy_name="fixed_512")
    gold_spans = [{"doc_id": "hr_leave_policy", "start": 103, "end": 151}]
    ids = gold_span_to_chunk_ids(gold_spans, chunks)
    assert len(ids) >= 1
    assert all(i.startswith("hr_leave_policy::span:") for i in ids)


def test_retrieval_off_returns_empty():
    index = RAGIndex.build(strategy_name="fixed_256", embeddings_kind="hash")
    docs = retrieve("anything", index, RetrieveConfig(retrieval_enabled=False))
    assert docs == []


def test_corpus_min_size():
    docs = load_corpus()
    assert len(docs) >= 8
