"""python -m rag_bench  →  same as run_all (one recompute command)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from rag_bench.run_all import main as run_all_main

    return run_all_main(argv)


if __name__ == "__main__":
    sys.exit(main())
