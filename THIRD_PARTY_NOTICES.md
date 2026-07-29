# Third-party notices

This project depends on open-source libraries declared in `pyproject.toml`.
Runtime and optional dependencies (not exhaustive; pin versions via your lock/environment):

## Core (runtime)

| Package | Typical license | Role |
| --- | --- | --- |
| langchain / langchain-core / langchain-community / langchain-text-splitters | MIT | RAG orchestration helpers, text splitters |
| langgraph / langgraph-checkpoint-sqlite | MIT | Stateful graph + SQLite checkpointer |
| faiss-cpu | MIT | Vector index |
| rank-bm25 | Apache-2.0 | BM25 sparse retrieval |
| numpy | BSD | Numerics |
| PyYAML | MIT | Config load |
| pydantic | MIT | Validation (transitive / API models) |

## Optional

| Package | Typical license | Role |
| --- | --- | --- |
| sentence-transformers | Apache-2.0 | MiniLM dense embeddings (`[semantic]` extra) |
| pytest | MIT | Tests (`[dev]` extra) |

## Models (optional download)

When using the MiniLM path, `sentence-transformers` may download:

- `sentence-transformers/all-MiniLM-L6-v2` (see model card on Hugging Face for license/terms)

No model weights are vendored in this repository.

## Notice

Upstream license texts remain with their packages as installed by pip.
This file is an informational inventory, not legal advice.
If you redistribute binaries or wheels that bundle dependencies, include the corresponding notices from those packages.
