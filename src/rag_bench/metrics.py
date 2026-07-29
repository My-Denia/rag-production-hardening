"""External-judge metrics: Recall@K, MRR, attribution suite, refusal metrics."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from langchain_core.documents import Document

from rag_bench.chunking import gold_span_to_chunk_ids, spans_overlap

ABSTAIN_MARK = "[ABSTAIN]"


def parse_chunk_id(chunk_id: str) -> tuple[str, int, int] | None:
    if "::span:" not in chunk_id:
        return None
    doc_id, rest = chunk_id.split("::span:", 1)
    try:
        start_s, end_s = rest.split(":", 1)
        return doc_id, int(start_s), int(end_s)
    except ValueError:
        return None


def chunk_overlaps_gold_spans(chunk_id: str, gold_spans: Sequence[dict]) -> bool:
    parsed = parse_chunk_id(chunk_id)
    if not parsed:
        return False
    doc_id, c_start, c_end = parsed
    for span in gold_spans:
        if span.get("doc_id") != doc_id:
            continue
        if spans_overlap(c_start, c_end, int(span["start"]), int(span["end"])):
            return True
    return False


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_spans: Sequence[dict],
    gold_doc_ids: Sequence[str] | None = None,
) -> float:
    if gold_spans:
        for cid in retrieved_chunk_ids:
            if chunk_overlaps_gold_spans(cid, gold_spans):
                return 1.0
        return 0.0
    if gold_doc_ids:
        gold = set(gold_doc_ids)
        for cid in retrieved_chunk_ids:
            doc_id = cid.split("::", 1)[0]
            if doc_id in gold:
                return 1.0
        return 0.0
    return 0.0


def first_gold_rank(
    retrieved_chunk_ids: Sequence[str],
    gold_spans: Sequence[dict],
    gold_doc_ids: Sequence[str] | None = None,
) -> int | None:
    """1-based rank of first gold-overlapping chunk, or None."""
    for i, cid in enumerate(retrieved_chunk_ids, start=1):
        if gold_spans:
            if chunk_overlaps_gold_spans(cid, gold_spans):
                return i
        elif gold_doc_ids:
            if cid.split("::", 1)[0] in set(gold_doc_ids):
                return i
    return None


def mrr_score(
    retrieved_chunk_ids: Sequence[str],
    gold_spans: Sequence[dict],
    gold_doc_ids: Sequence[str] | None = None,
) -> float:
    rank = first_gold_rank(retrieved_chunk_ids, gold_spans, gold_doc_ids)
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def attribution_score(
    source_chunk_ids: Sequence[str],
    answer: str,
    gold_spans: Sequence[dict],
    must_contain: Sequence[str],
) -> float:
    span_hit = any(chunk_overlaps_gold_spans(cid, gold_spans) for cid in source_chunk_ids)
    must_ok = all(token in answer for token in must_contain) if must_contain else True
    if span_hit and must_ok:
        return 1.0
    if span_hit or must_ok:
        return 0.5
    return 0.0


def attribution_hit(
    source_chunk_ids: Sequence[str],
    answer: str,
    gold_spans: Sequence[dict],
    must_contain: Sequence[str],
) -> float:
    """Binary attribution for selection (cited gold overlap AND all must_contain)."""
    span_hit = any(chunk_overlaps_gold_spans(cid, gold_spans) for cid in source_chunk_ids)
    must_ok = all(token in answer for token in must_contain) if must_contain else True
    return 1.0 if (span_hit and must_ok) else 0.0


def is_refused(answer: str, stage: str | None = None) -> bool:
    if (answer or "").strip() == ABSTAIN_MARK:
        return True
    if stage and "abstain" in str(stage).lower():
        return True
    return False


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def evaluate_run(
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    records items:
      retrieved_chunk_ids, source_chunk_ids, answer, gold_spans, must_contain,
      gold_doc_ids, answerable (optional), stage (optional)
    """
    recalls = []
    recalls_at_1 = []
    mrrs = []
    attrs = []
    attr_prec = []
    attr_rec = []
    attr_exact = []
    completeness = []
    unsupported = []
    error_cite = []
    hit_vector: list[int] = []
    attr_vector: list[int] = []

    n_refused = 0
    n_unanswerable = 0
    n_refused_and_unans = 0
    n_answerable = 0

    for r in records:
        answerable = r.get("answerable")
        if answerable is None:
            # legacy: treat as answerable if gold_spans or must_contain present
            answerable = bool(r.get("gold_spans") or r.get("must_contain") or r.get("gold_doc_ids"))
        answerable = bool(answerable)
        answer = r.get("answer") or ""
        stage = r.get("stage")
        refused = is_refused(answer, stage)
        gold_spans = r.get("gold_spans") or []
        gold_docs = r.get("gold_doc_ids")
        must = r.get("must_contain") or []
        retrieved = r.get("retrieved_chunk_ids") or []
        sources = r.get("source_chunk_ids") or []

        if not answerable:
            n_unanswerable += 1
            if refused:
                n_refused_and_unans += 1
                n_refused += 1
            else:
                if refused:
                    n_refused += 1
                # unsupported if non-empty answer and not refused
                if answer.strip() and not refused:
                    unsupported.append(1.0)
                else:
                    unsupported.append(0.0)
            continue

        n_answerable += 1
        if refused:
            n_refused += 1

        hit = recall_at_k(retrieved, gold_spans, gold_docs)
        recalls.append(hit)
        hit_vector.append(int(hit))
        r1 = recall_at_k(retrieved[:1], gold_spans, gold_docs) if retrieved else 0.0
        recalls_at_1.append(r1)
        mrrs.append(mrr_score(retrieved, gold_spans, gold_docs))

        ah = attribution_hit(sources, answer, gold_spans, must)
        attrs.append(ah)
        attr_vector.append(int(ah))

        cited_gold = [cid for cid in sources if chunk_overlaps_gold_spans(cid, gold_spans)]
        attr_prec.append(len(cited_gold) / max(len(sources), 1))
        ar = 1.0 if any(chunk_overlaps_gold_spans(cid, gold_spans) for cid in sources) else 0.0
        attr_rec.append(ar)
        must_ok = all(t in answer for t in must) if must else True
        attr_exact.append(1.0 if (must_ok and ar == 1.0) else 0.0)
        if must:
            completeness.append(sum(1 for t in must if t in answer) / len(must))
        else:
            completeness.append(1.0 if ar == 1.0 else 0.0)

        # unsupported: answerable, non-empty, no gold-overlap cite
        if answer.strip() and ar == 0.0 and not refused:
            unsupported.append(1.0)
        else:
            unsupported.append(0.0)

        # error citation: any cited chunk with no gold overlap while answerable
        if sources and any(not chunk_overlaps_gold_spans(cid, gold_spans) for cid in sources):
            error_cite.append(1.0)
        else:
            error_cite.append(0.0)

    refusal_precision = (
        n_refused_and_unans / max(n_refused, 1) if n_refused else 0.0
    )
    # if no refusals, precision is 0 by formula max(count(refused),1) → 0/1 = 0
    if n_refused == 0:
        refusal_precision = 0.0
    else:
        refusal_precision = n_refused_and_unans / n_refused

    refusal_recall = n_refused_and_unans / max(n_unanswerable, 1) if n_unanswerable else 0.0
    if n_unanswerable == 0:
        refusal_recall = 0.0
    else:
        refusal_recall = n_refused_and_unans / n_unanswerable

    if refusal_precision + refusal_recall > 0:
        refusal_f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall)
    else:
        refusal_f1 = 0.0

    return {
        "n": len(records),
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "recall_at_k": mean(recalls),
        "recall_at_1": mean(recalls_at_1),
        "mrr": mean(mrrs),
        "attribution_rate": mean(attrs),
        "attribution_precision": mean(attr_prec),
        "attribution_recall": mean(attr_rec),
        "attribution_exactness": mean(attr_exact),
        "completeness": mean(completeness),
        "unsupported_answer_rate": mean(unsupported) if unsupported else 0.0,
        "error_citation_rate": mean(error_cite),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "refusal_f1": refusal_f1,
        "recall_hits": int(sum(recalls)),
        "attribution_hits": int(sum(attrs)),
        "hit_vector": hit_vector,
        "attr_vector": attr_vector,
    }


def map_gold_chunk_ids_for_chunks(
    gold_spans: Sequence[dict],
    chunks: Sequence[Document],
) -> list[str]:
    return gold_span_to_chunk_ids(list(gold_spans), list(chunks))
