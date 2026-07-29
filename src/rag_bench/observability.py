"""Lightweight JSONL traces for pipeline observability."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from rag_bench.config_load import RESULTS_DIR, ensure_dirs

TRACES_DIR = RESULTS_DIR / "traces"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def write_trace_event(
    trace_id: str,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> Path:
    ensure_dirs()
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    out = path or (TRACES_DIR / f"{trace_id}.jsonl")
    rec = {
        "ts": time.time(),
        "trace_id": trace_id,
        "event": event,
        **(payload or {}),
    }
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def write_run_trace(summary: dict[str, Any]) -> Path:
    ensure_dirs()
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    tid = summary.get("trace_id") or new_trace_id()
    path = TRACES_DIR / f"run_{tid}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
