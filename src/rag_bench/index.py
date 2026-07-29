"""Embeddings + FAISS / BM25 index builders (LangChain ecosystem)."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_bench.chunking import chunk_documents


class HashEmbeddings(Embeddings):
    """
    Deterministic local embeddings: bag-of-hashed-tokens → fixed dim L2-normalized vector.
    Offline, no network, LangChain Embeddings-compatible.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = np.zeros(self.dim, dtype=np.float64)
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            # Use several bytes to place energy in multiple dims.
            for i in range(0, 16, 4):
                idx = int.from_bytes(h[i : i + 2], "little") % self.dim
                sign = 1.0 if h[i + 2] % 2 == 0 else -1.0
                weight = 1.0 + (h[i + 3] / 255.0)
                vec[idx] += sign * weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embeddings(kind: str = "hash", corpus_texts: Sequence[str] | None = None) -> Embeddings:
    """
    embeddings kinds:
      - hash: deterministic bag-of-hashed-tokens (non-semantic baseline)
      - tfidf: content-aware lexical TF-IDF (NOT semantic)
      - minilm: sentence-transformers MiniLM (semantic; may fail offline)
    """
    k = (kind or "hash").lower()
    if k in ("hash", "fake", "deterministic"):
        return HashEmbeddings(dim=256)
    if k in ("tfidf", "content_aware_lexical"):
        from rag_bench.embeddings_tfidf import TfidfEmbeddings

        emb = TfidfEmbeddings(dim=256)
        if corpus_texts:
            emb.fit(corpus_texts)
        return emb
    if k in ("minilm", "semantic", "all-minilm-l6-v2"):
        from rag_bench.embeddings_minilm import MiniLMEmbeddings

        return MiniLMEmbeddings()
    raise ValueError(f"Unsupported embeddings kind: {kind}")


class BM25Index:
    """Minimal BM25 over Document chunks (uses rank_bm25)."""

    def __init__(self, documents: Sequence[Document]):
        from rank_bm25 import BM25Okapi

        self.documents = list(documents)
        self._tokenized = [self._tokenize(d.page_content) for d in self.documents]
        self._bm25 = BM25Okapi(self._tokenized) if self.documents else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        if not self.documents or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        out: list[tuple[Document, float]] = []
        for i in order:
            s = float(scores[i])
            if s <= 0 and len(out) > 0:
                continue
            out.append((self.documents[int(i)], s))
        return out


class RAGIndex:
    """Holds dense FAISS store + BM25 over the same chunk set."""

    def __init__(
        self,
        chunks: list[Document],
        embeddings: Embeddings | None = None,
        embeddings_kind: str = "hash",
    ):
        self.chunks = chunks
        texts = [c.page_content for c in chunks]
        if embeddings is None:
            self.embeddings = get_embeddings(embeddings_kind, corpus_texts=texts)
        else:
            self.embeddings = embeddings
            # Fit TF-IDF if needed and not yet fitted
            if embeddings_kind in ("tfidf", "content_aware_lexical"):
                fit = getattr(self.embeddings, "fit", None)
                n_docs = getattr(self.embeddings, "n_docs", 0)
                if callable(fit) and not n_docs and texts:
                    fit(texts)
        self.embeddings_kind = embeddings_kind
        metadatas = [dict(c.metadata) for c in chunks]
        if texts:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            # Empty index edge case
            self.vectorstore = FAISS.from_texts([" "], self.embeddings, metadatas=[{"doc_id": "_empty"}])
            self.chunks = []
        self.bm25 = BM25Index(self.chunks)

    @classmethod
    def build(
        cls,
        strategy_name: str = "fixed_512",
        embeddings_kind: str = "hash",
    ) -> "RAGIndex":
        chunks = chunk_documents(strategy_name=strategy_name)
        return cls(chunks, embeddings_kind=embeddings_kind)

    def dense_search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        if not self.chunks:
            return []
        pairs = self.vectorstore.similarity_search_with_score(query, k=k)
        # FAISS L2 distance: convert to similarity-like score (higher better)
        out: list[tuple[Document, float]] = []
        for doc, dist in pairs:
            sim = 1.0 / (1.0 + float(dist))
            out.append((doc, sim))
        return out

    def sparse_search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        return self.bm25.search(query, k=k)

    def hybrid_search(
        self, query: str, k: int = 4, rrf_k: int = 60
    ) -> list[tuple[Document, float]]:
        """Hybrid dense+BM25 fused with Reciprocal Rank Fusion (rrf_k default 60)."""
        dense = self.dense_search(query, k=max(k * 2, k))
        sparse = self.sparse_search(query, k=max(k * 2, k))
        return rrf_fuse([dense, sparse], k=k, rrf_k=rrf_k)


def rrf_fuse(
    ranked_lists: list[list[tuple[Document, float]]],
    *,
    k: int = 4,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse multiple ranked (doc, score) lists via RRF: sum 1/(rrf_k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for pairs in ranked_lists:
        for rank, (doc, _) in enumerate(pairs):
            cid = str(doc.metadata.get("chunk_id", id(doc)))
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (float(rrf_k) + rank)
            docs[cid] = doc
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(docs[cid], sc) for cid, sc in ordered]
