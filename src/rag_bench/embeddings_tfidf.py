"""Content-aware lexical embeddings (TF-IDF). Not semantic — offline pure numpy."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfEmbeddings(Embeddings):
    """
    Fit-on-corpus TF-IDF bag-of-words projected to fixed dim via hashing trick.
    Content-aware lexical (AC4 fallback label: content_aware_lexical), NOT semantic.
    """

    def __init__(self, dim: int = 256, corpus_texts: Sequence[str] | None = None):
        self.dim = dim
        self.df: Counter[str] = Counter()
        self.n_docs = 0
        self.idf: dict[str, float] = {}
        if corpus_texts:
            self.fit(corpus_texts)

    def fit(self, texts: Iterable[str]) -> "TfidfEmbeddings":
        docs = list(texts)
        self.n_docs = max(len(docs), 1)
        self.df = Counter()
        for t in docs:
            toks = set(_tokenize(t))
            for tok in toks:
                self.df[tok] += 1
        self.idf = {
            tok: math.log((1.0 + self.n_docs) / (1.0 + df)) + 1.0
            for tok, df in self.df.items()
        }
        return self

    def _embed_one(self, text: str) -> list[float]:
        toks = _tokenize(text)
        if not toks:
            toks = ["empty"]
        tf = Counter(toks)
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok, cnt in tf.items():
            idf = self.idf.get(tok, math.log((1.0 + self.n_docs) / 1.0) + 1.0)
            weight = (cnt / max(len(toks), 1)) * idf
            # Hashing trick with sign
            h = abs(hash(tok))
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign * weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
