"""Retrieval stage with threshold filtering, hybrid RRF, multi-query static."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from rag_bench.index import RAGIndex, rrf_fuse


@dataclass
class RetrieveConfig:
    top_k: int = 4
    threshold: float | None = None
    retriever: str = "dense"  # dense | bm25 | hybrid | hybrid_rrf
    retrieval_enabled: bool = True
    rrf_k: int = 60
    query_strategy: str = "raw"  # raw | multi_query_static


def apply_threshold(
    pairs: list[tuple[Document, float]],
    threshold: float | None,
) -> list[tuple[Document, float]]:
    if threshold is None:
        return pairs
    return [(d, s) for d, s in pairs if s >= float(threshold)]


def _search_once(
    query: str,
    index: RAGIndex,
    cfg: RetrieveConfig,
    k: int,
) -> list[tuple[Document, float]]:
    retr = (cfg.retriever or "dense").lower()
    if retr in ("bm25", "sparse"):
        return index.sparse_search(query, k=k)
    if retr in ("hybrid", "hybrid_rrf", "rrf"):
        return index.hybrid_search(query, k=k, rrf_k=int(cfg.rrf_k))
    return index.dense_search(query, k=k)


def retrieve(
    query: str,
    index: RAGIndex,
    config: RetrieveConfig | None = None,
) -> list[Document]:
    """Return ranked documents for query (empty if retrieval disabled)."""
    cfg = config or RetrieveConfig()
    if not cfg.retrieval_enabled:
        return []
    k = max(1, int(cfg.top_k))
    fetch_k = max(k * 2, k)

    if (cfg.query_strategy or "raw") == "multi_query_static":
        from rag_bench.static_synonyms import expand_queries

        subqs = expand_queries(query, max_extra=2)
        lists: list[list[tuple[Document, float]]] = []
        for sq in subqs:
            lists.append(_search_once(sq, index, cfg, fetch_k))
        pairs = rrf_fuse(lists, k=k, rrf_k=int(cfg.rrf_k))
    else:
        pairs = _search_once(query, index, cfg, fetch_k)
        pairs = pairs[:k]

    pairs = apply_threshold(pairs, cfg.threshold)
    pairs = pairs[:k]
    docs: list[Document] = []
    for doc, score in pairs:
        md = dict(doc.metadata)
        md["score"] = float(score)
        docs.append(Document(page_content=doc.page_content, metadata=md))
    return docs


def docs_to_state(docs: list[Document]) -> list[dict[str, Any]]:
    """Serialize documents for LangGraph state / hashing."""
    out = []
    for d in docs:
        out.append(
            {
                "page_content": d.page_content,
                "chunk_id": d.metadata.get("chunk_id"),
                "doc_id": d.metadata.get("doc_id"),
                "start": d.metadata.get("start"),
                "end": d.metadata.get("end"),
                "score": d.metadata.get("score"),
            }
        )
    return out


def state_to_docs(items: list[dict[str, Any]] | None) -> list[Document]:
    if not items:
        return []
    docs = []
    for it in items:
        docs.append(
            Document(
                page_content=it.get("page_content", ""),
                metadata={
                    k: it.get(k)
                    for k in ("chunk_id", "doc_id", "start", "end", "score", "strategy")
                    if it.get(k) is not None
                },
            )
        )
    return docs
