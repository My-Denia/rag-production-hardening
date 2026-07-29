"""LangChain-compatible reranker (term-overlap compressor; offline)."""

from __future__ import annotations

import re
from typing import Optional, Sequence

from langchain_core.documents import BaseDocumentCompressor, Document


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap_score(query: str, doc_text: str) -> float:
    q = _tokens(query)
    d = _tokens(doc_text)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    return inter / max(len(q), 1)


class TermOverlapReranker(BaseDocumentCompressor):
    """
    Re-scores documents by query-term overlap.
    Implements LangChain BaseDocumentCompressor for stack fidelity without cross-encoder.
    """

    top_n: int = 4

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        scored: list[tuple[float, Document]] = []
        for doc in documents:
            s = overlap_score(query, doc.page_content)
            md = dict(doc.metadata)
            md["rerank_score"] = s
            # Blend with prior score if present
            prior = float(md.get("score") or 0.0)
            md["score"] = 0.6 * s + 0.4 * prior
            scored.append((md["score"], Document(page_content=doc.page_content, metadata=md)))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_n = self.top_n if self.top_n > 0 else len(scored)
        return [d for _, d in scored[:top_n]]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        return self.compress_documents(documents, query, callbacks=callbacks)


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_n: Optional[int] = None,
    enabled: bool = True,
) -> list[Document]:
    if not enabled:
        return list(documents)
    n = top_n if top_n is not None else len(documents)
    compressor = TermOverlapReranker(top_n=n)
    return list(compressor.compress_documents(list(documents), query))
