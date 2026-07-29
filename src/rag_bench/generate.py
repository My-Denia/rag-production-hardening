"""Extractive cite-based answer generation (offline default path)."""

from __future__ import annotations

import re
from typing import Any, Sequence

from langchain_core.documents import Document

from rag_bench.rerank import overlap_score


_TIME_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{1,2}:\d{2}|\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)|"
    r"\d+(?:\.\d+)?%?)\b",
    re.I,
)
_NUM_RE = re.compile(r"\b\d+\b")
_DISTRACTOR_RE = re.compile(r"^\s*\[|cross-ref note|not authoritative|ignore handbook", re.I)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _stem_token(tok: str) -> str:
    """Aggressive light stem for offline lexical match (not gold-informed)."""
    t = tok.lower()
    for suf in ("ations", "ation", "tions", "tion", "ings", "ing", "ers", "ies", "ied", "ed", "es", "s", "ly"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            t = t[: -len(suf)]
            break
    return t


def _tokens_stemmed(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out: set[str] = set()
    for t in raw:
        out.add(t)
        out.add(_stem_token(t))
        # prefix keys for soft morphology (password/passwords, rotate/rotation)
        if len(t) >= 5:
            out.add(t[:5])
    return out


def _stem_overlap(query: str, sent: str) -> float:
    q = _tokens_stemmed(query)
    d = _tokens_stemmed(sent)
    if not q or not d:
        return 0.0
    # drop ultra-common stop stems from query for scoring
    stop = {
        "what", "when", "where", "which", "who", "how", "the", "a", "an", "is", "are",
        "do", "does", "did", "to", "of", "and", "or", "for", "in", "on", "at", "by",
        "with", "from", "be", "been", "was", "were", "this", "that", "it", "its",
    }
    q_content = {t for t in q if t not in stop and len(t) > 1}
    if not q_content:
        q_content = q
    inter = len(q_content & d)
    return inter / max(len(q_content), 1)


def _query_wants_numeric_or_time(query: str) -> bool:
    ql = query.lower()
    keys = (
        "how many", "how long", "minimum", "maximum", "length", "period", "when",
        "start", "hours", "days", "weeks", "years", "limit", "cap", "rate",
        "discount", "stipend", "allowance", "sla", "uptime", "password", "rotation",
    )
    return any(k in ql for k in keys)


def sentence_score(query: str, sent: str, *, doc_rank: int = 0, doc_score: float = 0.0) -> float:
    """
    Composite extractive score: stem-aware overlap + fact density + rank prior.
    No gold/label inputs.
    """
    base = max(overlap_score(query, sent), _stem_overlap(query, sent))
    # Prefer earlier retrieved docs slightly
    rank_boost = max(0.0, 0.12 * (1.0 - min(doc_rank, 8) / 8.0))
    prior = 0.05 * float(doc_score or 0.0)

    fact_boost = 0.0
    if _query_wants_numeric_or_time(query):
        if _NUM_RE.search(sent):
            fact_boost += 0.15
        if _TIME_RE.search(sent):
            fact_boost += 0.12
        # times like 09:00
        if re.search(r"\d{1,2}:\d{2}", sent):
            fact_boost += 0.08

    penalty = 0.0
    if _DISTRACTOR_RE.search(sent):
        penalty += 0.35
    # Cross-doc rumor lines that mention "unrelated" / "not"
    if re.search(r"\bunrelated\b|\bmyths?\b|\brumors?\b", sent, re.I):
        penalty += 0.2

    return base + rank_boost + prior + fact_boost - penalty


def extractive_answer(
    query: str,
    documents: Sequence[Document],
    max_sentences: int = 4,
) -> dict[str, Any]:
    """
    Select top sentences by stem-aware overlap + fact density; cite source chunk_ids.
    Returns {answer, source_chunk_ids}.
    """
    if not documents:
        return {
            "answer": "No supporting context was retrieved.",
            "source_chunk_ids": [],
        }

    candidates: list[tuple[float, str, str]] = []
    for rank, doc in enumerate(documents):
        cid = str(doc.metadata.get("chunk_id") or doc.metadata.get("doc_id") or "unknown")
        prior = float(doc.metadata.get("score") or 0.0)
        for sent in _split_sentences(doc.page_content):
            if len(sent) < 12:
                continue
            s = sentence_score(query, sent, doc_rank=rank, doc_score=prior)
            candidates.append((s, sent, cid))

    candidates.sort(key=lambda x: x[0], reverse=True)

    answer_parts: list[str] = []
    source_ids: list[str] = []
    seen_sent: set[str] = set()
    # Keep weak-but-positive candidates if nothing better; avoid zero-score filler once we have content
    min_keep = 0.05
    for score, sent, cid in candidates:
        if score < min_keep and answer_parts:
            break
        key = sent.lower()
        if key in seen_sent:
            continue
        seen_sent.add(key)
        answer_parts.append(f"{sent} [cite:{cid}]")
        if cid not in source_ids:
            source_ids.append(cid)
        if len(answer_parts) >= max_sentences:
            break

    if not answer_parts:
        doc0 = documents[0]
        cid = str(doc0.metadata.get("chunk_id") or "unknown")
        text = doc0.page_content.strip().replace("\n", " ")
        return {
            "answer": text[:400] + f" [cite:{cid}]",
            "source_chunk_ids": [cid],
        }

    return {
        "answer": " ".join(answer_parts),
        "source_chunk_ids": source_ids,
    }
