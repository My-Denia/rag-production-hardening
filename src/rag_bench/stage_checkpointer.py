"""Checkpointer wrapper: durable put then stage sidecar + optional post-stage hold."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


class StageAwareCheckpointer:
    """
    Wraps a LangGraph checkpointer (e.g. SqliteSaver).

    After each successful durable put:
      1. commit SQLite if present
      2. write atomic stage sidecar from channel_values
      3. if stage in hold_stages and hold_sec>0, sleep once per stage (test kill window)

    Hold runs AFTER durable write — not a LangGraph interrupt_after.
    """

    def __init__(
        self,
        inner: Any,
        stage_path: str | Path,
        hold_sec: float | None = None,
        hold_stages: list[str] | None = None,
    ):
        self._inner = inner
        self.stage_path = Path(stage_path)
        if hold_sec is None:
            hold_sec = float(os.environ.get("RAG_POST_RETRIEVE_HOLD_SEC", "0") or "0")
        self.hold_sec = float(hold_sec)
        if hold_stages is None:
            env_stages = os.environ.get("RAG_HOLD_STAGES", "retrieve").strip()
            hold_stages = [s.strip() for s in env_stages.split(",") if s.strip()]
        self.hold_stages = set(hold_stages or ["retrieve"])
        self._held: set[str] = set()

    def put(self, config, checkpoint, metadata, new_versions):
        result = self._inner.put(config, checkpoint, metadata, new_versions)
        conn = getattr(self._inner, "conn", None)
        if conn is not None:
            try:
                conn.commit()
            except Exception:
                pass

        values = {}
        if isinstance(checkpoint, dict):
            values = checkpoint.get("channel_values") or {}
        else:
            values = getattr(checkpoint, "channel_values", None) or {}
            if not isinstance(values, dict):
                values = dict(values) if values else {}

        stage = values.get("stage")
        answer = values.get("answer") or ""
        retrieved = values.get("retrieved_docs") or []
        answer_empty = not bool(str(answer).strip())

        from rag_bench.stage_sidecar import write_stage_sidecar

        write_stage_sidecar(
            self.stage_path,
            stage=stage,
            answer_empty=answer_empty,
            n_retrieved=len(retrieved) if isinstance(retrieved, list) else 0,
            extra={"hold_sec": self.hold_sec, "held_stages": sorted(self._held)},
        )

        if (
            stage in self.hold_stages
            and self.hold_sec > 0
            and stage not in self._held
        ):
            # For retrieve, prefer empty answer (pre-generate); for later stages allow answer.
            if stage == "retrieve" and not answer_empty:
                return result
            self._held.add(str(stage))
            write_stage_sidecar(
                self.stage_path,
                stage=stage,
                answer_empty=answer_empty,
                n_retrieved=len(retrieved) if isinstance(retrieved, list) else 0,
                extra={"holding": True, "hold_sec": self.hold_sec, "hold_stage": stage},
            )
            time.sleep(self.hold_sec)

        return result

    def __getattr__(self, name: str):
        return getattr(self._inner, name)
