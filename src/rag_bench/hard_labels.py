"""Uniform difficulty bump: paraphrase questions + append distractors (preserve gold offsets)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag_bench.config_load import CORPUS_DIR, DATA_DIR, LABELS_PATH
from rag_bench.eval import load_labels

# Paraphrases for v1 questions (qid-aligned). Gold spans / must_contain unchanged.
PARAPHRASES_V2: dict[str, str] = {
    "q01": "For staff under five years tenure, what yearly vacation allotment applies in working days?",
    "q02": "After completing five years of continuous employment, what yearly vacation allotment applies?",
    "q03": "What ceiling applies when rolling unused vacation into the following calendar year?",
    "q04": "State the paid duration of parental leave for birth mothers and for fathers at this firm.",
    "q05": "Must secondary authentication factors be enabled when connecting via remote network tunnel or corporate mail?",
    "q06": "State the character floor for credentials and how frequently they must be changed.",
    "q07": "Is removable flash media permitted on corporate desktops without exception?",
    "q08": "State the reporting deadline and inbox for suspected credential phishing or malicious software.",
    "q09": "Above what amount must trips be authorized before incurring business trip costs?",
    "q10": "Without itemized meal tickets, what daily food stipend applies on trips inside the country?",
    "q11": "What daily food stipend applies when traveling outside the country?",
    "q12": "How soon after returning from a trip must cost filings be completed?",
    "q13": "On which weekday and UTC clock time does the main pager duty cycle begin?",
    "q14": "What update cadence is required in the incidents chat during Severity-1 events?",
    "q15": "State weekly stipends for main and backup pager duty rotations.",
    "q16": "After a Severity-1 outage, within how many business days must the post-incident review appear?",
    "q17": "Contrast Standard and Enterprise retention windows for Nova stored data.",
    "q18": "Describe the free evaluation period length and its data-source ceiling for Nova.",
    "q19": "What monthly availability target is promised for Nova Enterprise?",
    "q20": "Which offering handles protected health information, and should Nova be used for that workload?",
    "q21": "How long does Helix keep audit trail records?",
    "q22": "State Helix request-rate caps for Standard tenants versus Enterprise tenants.",
    "q23": "Which garage provides complimentary staff vehicle spaces at HQ?",
    "q24": "During which clock windows does the HQ canteen serve morning and midday meals?",
    "q25": "Where should personnel gather outdoors after an emergency building exit?",
    "q26": "For material that is not a trade secret, how long do mutual NDA secrecy duties continue?",
    "q27": "Which U.S. state's statutes control the mutual nondisclosure agreement?",
    "q28": "What price reduction may sales reps grant before needing higher-level sign-off?",
    "q29": "What multi-year loyalty rebate applies to three-year agreements?",
    "q30": "What standard sector price reduction applies to nonprofits and schools?",
    "q31": "How many weekly in-office days are mandatory for hybrid staff?",
    "q32": "What monthly broadband subsidy applies to fully remote staff?",
    "q33": "Between which local times must remote staff remain available for collaboration?",
    "q34": "For remote work abroad lasting more than two weeks straight, which groups must pre-approve?",
    "q35": "Contrast in-country versus cross-border daily food stipends for business trips.",
}

# Keyword-rich distractor blocks appended to non-gold docs (offsets of existing gold spans preserved).
DISTRACTOR_BLOCKS: list[tuple[str, str]] = [
    (
        "finance_expenses",
        "\n\n[Cross-ref note — not authoritative for leave] Some teams incorrectly quote "
        "annual paid leave as 20 or 25 days and MFA mandatory password rules of 14 characters "
        "every 90 days; those figures belong in HR/IT policies, not expense forms. "
        "Domestic meal myths of $45 and international $75 sometimes appear in sales decks "
        "without finance approval. Nova Enterprise 99.9% uptime is a product claim, not a travel rule.\n",
    ),
    (
        "sales_discounting",
        "\n\n[Cross-ref note — not pricing policy] Ignore handbook rumors that hybrid staff need "
        "3 office days, remote stipends of $50, or Helix audit logs for 7 years — those are not "
        "discount rules. Account executives sometimes confuse 10% self-serve discounts with "
        "leave carry-over of 5 days or on-call pay of $150/$50.\n",
    ),
    (
        "it_security",
        "\n\n[Cross-ref note — not security controls] Facilities trivia such as Garage B parking, "
        "cafeteria 07:30–11:30, or south plaza assembly is not an access-control requirement. "
        "Expense pre-approval at $500 and Delaware governing law are outside IT scope. "
        "Nova free trial of 14 days with 2 data sources is a product marketing line.\n",
    ),
    (
        "hr_leave_policy",
        "\n\n[Cross-ref note — not leave policy] Do not use this page for VPN MFA, USB blocked "
        "defaults, phishing report within 1 hour to security@aetherdynamics.example, "
        "or Helix API rates of 1000/10000. Those topics live elsewhere. "
        "Sector nonprofit discounts of 15% are sales matters.\n",
    ),
    (
        "hr_remote_work",
        "\n\n[Cross-ref note — not remote-work policy] Pager rotation starting Monday 09:00 UTC, "
        "Severity-1 updates every 30 minutes, and PIR within 5 business days are engineering "
        "on-call rules. Maternity 16 weeks / paternity 2 weeks remain leave-policy items. "
        "Loyalty credit of 5% on 3-year deals is commercial, not remote work.\n",
    ),
    (
        "eng_oncall",
        "\n\n[Cross-ref note — not on-call] Core collaboration hours 10:00–15:00 and Legal/Tax "
        "pre-approval for international remote beyond 14 days are HR remote rules. "
        "Confidentiality lasting 3 years under Delaware law is legal NDA text. "
        "Helix PHI handling differs from Nova Standard 30-day versus Enterprise 365-day retention.\n",
    ),
    (
        "product_nova",
        "\n\n[Cross-ref note — not Nova SKU] Helix retains audit logs 7 years and targets healthcare PHI; "
        "do not conflate with Nova trial limits. Expense submission within 30 days and password "
        "rotation every 90 days are unrelated operational policies. Garage B free parking is facilities.\n",
    ),
    (
        "product_helix",
        "\n\n[Cross-ref note — not Helix SKU] Nova Enterprise monthly uptime 99.9% and Standard "
        "retention of 30 days versus 365 for Enterprise are Nova numbers. On-call primary compensation "
        "of $150/week is engineering. Multi-factor authentication for VPN is IT security.\n",
    ),
    (
        "facilities_hq",
        "\n\n[Cross-ref note — not facilities] AE-approved discounts up to 10%, nonprofit 15% sector "
        "rates, and 5% multi-year loyalty credits are sales. USB mass storage blocked by default is IT. "
        "Annual leave carry-over max 5 days is HR leave policy.\n",
    ),
    (
        "legal_nda",
        "\n\n[Cross-ref note — not NDA] South plaza evacuation assembly, cafeteria breakfast 07:30, "
        "and home internet stipend $50 are not contractual NDA terms. Severity-1 channel updates "
        "every 30 minutes are operational. Travel pre-approval above $500 is finance.\n",
    ),
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def mean_query_gold_jaccard(
    labels: list[dict[str, Any]],
    corpus: dict[str, str],
) -> float:
    scores = []
    for lab in labels:
        q = _tokens(lab["question"])
        gold_toks: set[str] = set()
        for span in lab.get("gold_spans") or []:
            doc = corpus.get(span["doc_id"], "")
            gold_toks |= _tokens(doc[int(span["start"]) : int(span["end"])])
        if not q and not gold_toks:
            scores.append(0.0)
            continue
        inter = len(q & gold_toks)
        union = len(q | gold_toks) or 1
        scores.append(inter / union)
    return sum(scores) / len(scores) if scores else 0.0


def ensure_distractors_appended(corpus_dir: Path | None = None) -> dict[str, bool]:
    """Append distractor blocks once (idempotent via marker)."""
    root = corpus_dir or CORPUS_DIR
    changed: dict[str, bool] = {}
    for doc_id, block in DISTRACTOR_BLOCKS:
        path = root / f"{doc_id}.txt"
        if not path.exists():
            changed[doc_id] = False
            continue
        text = path.read_text(encoding="utf-8")
        marker = "[Cross-ref note"
        if marker in text:
            changed[doc_id] = False
            continue
        path.write_text(text.rstrip() + block, encoding="utf-8")
        changed[doc_id] = True
    return changed


def build_harder_labels(
    labels_v1: list[dict[str, Any]] | None = None,
    paraphrases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    labels_v1 = labels_v1 or load_labels(DATA_DIR / "labels_v1.jsonl")
    paraphrases = paraphrases or PARAPHRASES_V2
    out = []
    for lab in labels_v1:
        lab2 = dict(lab)
        qid = lab["qid"]
        if qid in paraphrases:
            lab2["question"] = paraphrases[qid]
        lab2["difficulty"] = "v2_hard"
        out.append(lab2)
    return out


def write_harder_labels(path: Path | None = None) -> Path:
    path = path or LABELS_PATH
    labels = build_harder_labels()
    with path.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    return path


def difficulty_gate_report(
    labels_v1: list[dict[str, Any]],
    labels_v2: list[dict[str, Any]],
    corpus: dict[str, str],
    *,
    v1_hash_recall: float,
    v2_hash_recall: float,
    min_recall_drop: float = 0.15,
) -> dict[str, Any]:
    j1 = mean_query_gold_jaccard(labels_v1, corpus)
    j2 = mean_query_gold_jaccard(labels_v2, corpus)
    drop = float(v1_hash_recall) - float(v2_hash_recall)
    return {
        "jaccard_v1": j1,
        "jaccard_v2": j2,
        "jaccard_drop": j1 - j2,
        "jaccard_strictly_lower": j2 < j1,
        "hash_recall_v1": v1_hash_recall,
        "hash_recall_v2": v2_hash_recall,
        "hash_recall_drop": drop,
        "hash_recall_drop_gate": drop >= min_recall_drop,
        "min_recall_drop_required": min_recall_drop,
        "passed": (j2 < j1) and (drop >= min_recall_drop),
    }
