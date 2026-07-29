"""Load YAML configs for ablation grid and selected params."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
LABELS_PATH = DATA_DIR / "labels.jsonl"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
CACHE_DIR = ROOT / ".cache"


def project_root() -> Path:
    return ROOT


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {p}")
    return data


def load_ablation_grid() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "ablation_grid.yaml")


def load_selected() -> dict[str, Any]:
    path = CONFIG_DIR / "selected.yaml"
    if not path.exists():
        # Conservative bootstrap defaults; overwritten by ablations in run_all.
        return {
            "chunk_strategy": "fixed_512",
            "top_k": 4,
            "rerank": True,
            "retrieval": True,
            "threshold": None,
            "embeddings": "hash",
            "retriever": "dense",
            "source": "bootstrap_default_pre_ablation",
        }
    return load_yaml(path)


def save_selected(cfg: dict[str, Any]) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / "selected.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
