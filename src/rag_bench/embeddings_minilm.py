"""Optional sentence-transformers MiniLM embeddings (true semantic path)."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings


class MiniLMEmbeddings(Embeddings):
    """
    Wrapper around sentence-transformers all-MiniLM-L6-v2.
    Import / model load may fail offline — callers must fail-soft.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed; MiniLM backend unavailable"
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        emb = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in emb]

    def embed_query(self, text: str) -> list[float]:
        emb = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return emb[0].tolist()


def try_load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> tuple[Any | None, dict]:
    """Attempt load; return (embeddings_or_None, report_dict)."""
    report: dict = {
        "backend": "minilm",
        "model_name": model_name,
        "attempted": True,
        "success": False,
        "error": None,
    }
    try:
        emb = MiniLMEmbeddings(model_name=model_name)
        # Tiny smoke encode
        _ = emb.embed_query("smoke test")
        report["success"] = True
        return emb, report
    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        return None, report
