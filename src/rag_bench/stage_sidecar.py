"""Atomic stage sidecar writes for process-kill recovery probes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def write_stage_sidecar(
    path: str | Path,
    *,
    stage: str | None,
    answer_empty: bool,
    n_retrieved: int = 0,
    pid: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically write stage JSON (tmp + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "stage": stage,
        "answer_empty": bool(answer_empty),
        "n_retrieved": int(n_retrieved),
        "pid": int(pid if pid is not None else os.getpid()),
        "ts": time.time(),
    }
    if extra:
        payload.update(extra)
    tmp = p.with_suffix(p.suffix + ".tmp")
    data = json.dumps(payload, ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def read_stage_sidecar(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
