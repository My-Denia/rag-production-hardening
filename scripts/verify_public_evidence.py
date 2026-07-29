#!/usr/bin/env python3
"""CLI entry for public evidence verifier (thin wrapper)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without install: add src/ to path
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rag_bench.verify_public_evidence import main

if __name__ == "__main__":
    raise SystemExit(main())
