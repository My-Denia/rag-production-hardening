"""Static domain synonym handbook for multi_query_static (NOT gold-informed)."""

from __future__ import annotations

import re

# Word/phrase replacements only — domain dictionary for Aether Dynamics handbook.
# Forbidden: reading gold spans/labels into expansions.
SYNONYM_TABLE: list[tuple[str, str]] = [
    (r"\bleave\b", "vacation"),
    (r"\bvacation\b", "leave"),
    (r"\bannual paid leave\b", "yearly vacation allotment"),
    (r"\byearly vacation allotment\b", "annual paid leave"),
    (r"\bpassword\b", "credential"),
    (r"\bcredential\b", "password"),
    (r"\bMFA\b", "multi-factor authentication"),
    (r"\bmulti-factor authentication\b", "MFA"),
    (r"\bVPN\b", "remote network tunnel"),
    (r"\bremote network tunnel\b", "VPN"),
    (r"\bSLA\b", "service level agreement"),
    (r"\bservice level agreement\b", "SLA"),
    (r"\buptime\b", "availability"),
    (r"\bavailability\b", "uptime"),
    (r"\breimbursement\b", "expense refund"),
    (r"\bexpense\b", "reimbursement"),
    (r"\bmeal allowance\b", "food stipend"),
    (r"\bfood stipend\b", "meal allowance"),
    (r"\bon-call\b", "pager duty"),
    (r"\bpager duty\b", "on-call"),
    (r"\bincident\b", "outage event"),
    (r"\bdiscount\b", "price reduction"),
    (r"\bprice reduction\b", "discount"),
    (r"\bNDA\b", "nondisclosure agreement"),
    (r"\bnondisclosure agreement\b", "NDA"),
    (r"\bconfidentiality\b", "secrecy obligation"),
    (r"\bparking\b", "vehicle spaces"),
    (r"\bhybrid\b", "office-remote mix"),
    (r"\bremote work\b", "work from home"),
    (r"\bstipend\b", "subsidy"),
    (r"\baudit logs\b", "audit trail records"),
    (r"\bretention\b", "data keep window"),
    (r"\bPHI\b", "protected health information"),
    (r"\bprotected health information\b", "PHI"),
    (r"\bemployee\b", "staff"),
    (r"\bstaff\b", "employee"),
    (r"\bmandatory\b", "required"),
    (r"\brequired\b", "mandatory"),
]


def expand_queries(raw: str, max_extra: int = 2) -> list[str]:
    """
    Return [raw] + up to max_extra deterministic synonym expansions.
    Word-level replacements from fixed SYNONYM_TABLE only.
    """
    out = [raw]
    seen = {raw.lower()}
    for pattern, repl in SYNONYM_TABLE:
        if len(out) >= 1 + max_extra:
            break
        try:
            candidate = re.sub(pattern, repl, raw, count=1, flags=re.IGNORECASE)
        except re.error:
            continue
        if candidate and candidate.lower() not in seen and candidate != raw:
            seen.add(candidate.lower())
            out.append(candidate)
    return out
