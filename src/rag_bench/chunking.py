"""Named chunk strategies using LangChain text splitters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from rag_bench.config_load import CORPUS_DIR


@dataclass(frozen=True)
class ChunkStrategy:
    name: str
    kind: str  # fixed | recursive
    chunk_size: int
    chunk_overlap: int


STRATEGIES: dict[str, ChunkStrategy] = {
    "fixed_256": ChunkStrategy("fixed_256", "fixed", 256, 32),
    "fixed_512": ChunkStrategy("fixed_512", "fixed", 512, 64),
    "recursive_400_80": ChunkStrategy("recursive_400_80", "recursive", 400, 80),
}


def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def load_corpus(corpus_dir: Path | None = None) -> list[Document]:
    """Load each .txt file as a source document with doc_id metadata."""
    root = corpus_dir or CORPUS_DIR
    docs: list[Document] = []
    for path in sorted(root.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": doc_id,
                    "source": str(path),
                    "source_text": text,
                },
            )
        )
    if not docs:
        raise FileNotFoundError(f"No corpus files under {root}")
    return docs


def _make_splitter(strategy: ChunkStrategy):
    if strategy.kind == "fixed":
        return CharacterTextSplitter(
            separator="\n",
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    if strategy.kind == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=strategy.chunk_size,
            chunk_overlap=strategy.chunk_overlap,
            length_function=len,
        )
    raise ValueError(f"Unknown strategy kind: {strategy.kind}")


def chunk_documents(
    documents: Iterable[Document] | None = None,
    strategy_name: str = "fixed_512",
) -> list[Document]:
    """
    Split corpus docs into chunks with stable span-based chunk_ids.

    chunk_id format: {doc_id}::span:{start}:{end}
    where start/end are character offsets into the original source text.
    """
    if strategy_name not in STRATEGIES:
        raise KeyError(f"Unknown chunk strategy: {strategy_name}. Known: {list_strategies()}")
    strategy = STRATEGIES[strategy_name]
    source_docs = list(documents) if documents is not None else load_corpus()
    splitter = _make_splitter(strategy)

    chunks: list[Document] = []
    for doc in source_docs:
        doc_id = doc.metadata["doc_id"]
        source_text = doc.metadata.get("source_text") or doc.page_content
        parts = splitter.split_text(doc.page_content)
        # Map each part to the earliest unused occurrence in source_text.
        search_from = 0
        for part in parts:
            if not part.strip():
                continue
            idx = source_text.find(part, search_from)
            if idx < 0:
                # Overlap / separator edge cases: search whole doc.
                idx = source_text.find(part)
            if idx < 0:
                # Fallback: approximate by cumulative search_from.
                idx = min(search_from, max(0, len(source_text) - len(part)))
            start = idx
            end = idx + len(part)
            search_from = max(search_from, start + 1)
            chunk_id = f"{doc_id}::span:{start}:{end}"
            chunks.append(
                Document(
                    page_content=part,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "start": start,
                        "end": end,
                        "strategy": strategy_name,
                        "source": doc.metadata.get("source", ""),
                    },
                )
            )
    return chunks


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def gold_span_to_chunk_ids(
    gold_spans: list[dict],
    chunks: list[Document],
    min_overlap: int = 1,
) -> list[str]:
    """Map gold char spans to overlapping chunk_ids (strategy-independent labels)."""
    hits: list[str] = []
    seen: set[str] = set()
    for span in gold_spans:
        doc_id = span["doc_id"]
        g_start = int(span["start"])
        g_end = int(span["end"])
        for ch in chunks:
            if ch.metadata.get("doc_id") != doc_id:
                continue
            c_start = int(ch.metadata["start"])
            c_end = int(ch.metadata["end"])
            if spans_overlap(g_start, g_end, c_start, c_end):
                overlap = min(g_end, c_end) - max(g_start, c_start)
                if overlap >= min_overlap:
                    cid = ch.metadata["chunk_id"]
                    if cid not in seen:
                        seen.add(cid)
                        hits.append(cid)
    return hits
