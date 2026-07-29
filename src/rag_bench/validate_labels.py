"""Validate labels.jsonl schema and gold span offsets against corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_bench.config_load import CORPUS_DIR, LABELS_PATH, project_root

REQUIRED_KEYS = {"qid", "question", "gold_doc_ids", "gold_spans", "must_contain"}
OPTIONAL_NEW = {
    "category",
    "answerable",
    "evidence_ids",
    "negative_docs",
    "refusal_expected",
}


def validate(labels_path: Path | None = None, corpus_dir: Path | None = None) -> int:
    labels_path = labels_path or LABELS_PATH
    corpus_dir = corpus_dir or CORPUS_DIR
    errors: list[str] = []
    warnings: list[str] = []

    corpus_files = {p.stem: p.read_text(encoding="utf-8") for p in corpus_dir.glob("*.txt")}
    if len(corpus_files) < 8:
        errors.append(f"Need ≥8 corpus docs, found {len(corpus_files)}")

    labels = []
    with labels_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON: {e}")
                continue
            labels.append((i, obj))

    if len(labels) < 15:
        errors.append(f"Need ≥15 labels, found {len(labels)}")

    qids = set()
    for i, lab in labels:
        missing = REQUIRED_KEYS - set(lab.keys())
        if missing:
            errors.append(f"line {i} qid={lab.get('qid')}: missing keys {missing}")
        qid = lab.get("qid")
        if qid in qids:
            errors.append(f"duplicate qid: {qid}")
        qids.add(qid)

        answerable = lab.get("answerable")
        if answerable is None:
            answerable = bool(lab.get("gold_spans") or lab.get("must_contain"))
        answerable = bool(answerable)

        if lab.get("category") == "unanswerable" or lab.get("refusal_expected") is True:
            if answerable and lab.get("category") == "unanswerable":
                errors.append(f"{qid}: unanswerable category must have answerable=false")
            if not answerable:
                if lab.get("gold_spans"):
                    errors.append(f"{qid}: unanswerable must have empty gold_spans")
                if lab.get("must_contain"):
                    errors.append(f"{qid}: unanswerable must have empty must_contain")

        gold_docs = lab.get("gold_doc_ids") or []
        for d in gold_docs:
            if d not in corpus_files:
                errors.append(f"{qid}: gold_doc_id not in corpus: {d}")

        for span in lab.get("gold_spans") or []:
            doc_id = span.get("doc_id")
            start = span.get("start")
            end = span.get("end")
            if doc_id not in corpus_files:
                errors.append(f"{qid}: span doc_id missing: {doc_id}")
                continue
            text = corpus_files[doc_id]
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(text) or start >= end:
                errors.append(f"{qid}: invalid span bounds {start}:{end} for {doc_id} (len={len(text)})")
                continue
            snippet = text[start:end]
            if not snippet.strip():
                errors.append(f"{qid}: empty gold span text")

        must = lab.get("must_contain") or []
        if answerable and not must:
            warnings.append(f"{qid}: empty must_contain")
        if answerable:
            for token in must:
                found = False
                for span in lab.get("gold_spans") or []:
                    doc_id = span.get("doc_id")
                    if doc_id not in corpus_files:
                        continue
                    if token in corpus_files[doc_id]:
                        found = True
                        break
                if not found and gold_docs:
                    for d in gold_docs:
                        if d in corpus_files and token in corpus_files[d]:
                            found = True
                            break
                if not found:
                    warnings.append(f"{qid}: must_contain token not found in gold docs: {token!r}")

        if lab.get("category") == "multi_evidence":
            eids = lab.get("evidence_ids") or []
            if len(eids) < 2:
                errors.append(f"{qid}: multi_evidence requires evidence_ids length ≥2")

        for nd in lab.get("negative_docs") or []:
            if nd not in corpus_files:
                warnings.append(f"{qid}: negative_doc not in corpus: {nd}")

    print(f"corpus_docs={len(corpus_files)} labels={len(labels)}")
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        print("validate_labels: FAIL")
        return 1
    print("validate_labels: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        return validate(Path(argv[0]))
    return validate()


if __name__ == "__main__":
    sys.exit(main())
