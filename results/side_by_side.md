# Side-by-side: run_v1 vs run_v2

## Metrics

| metric | v1 | v2 |
| --- | --- | --- |
| attribution_rate | 0.9714285714285714 | 0.6857142857142857 |
| chunk_strategy | fixed_256 | fixed_512 |
| embeddings | hash | minilm |
| n | 35 | 35 |
| recall_at_k | 0.9714285714285714 | 0.9714285714285714 |
| top_k | 2 | 8 |

## Factor discriminability claims

| factor | v1 | v2 |
| --- | --- | --- |
| chunk_strategy | non-discriminative | non-discriminative |
| embeddings | non-discriminative | non-discriminative |
| rerank | non-discriminative | non-discriminative |
| retrieval | discriminative | discriminative |
| threshold | non-discriminative | non-discriminative |
| top_k | non-discriminative | discriminative |

## Difficulty gates

- jaccard_v1: 0.3663448467501022
- jaccard_v2: 0.1236286354569305
- jaccard_drop: 0.24271621129317172
- hash_recall_v1: 0.9714285714285714
- hash_recall_v2: 0.5714285714285714
- hash_recall_drop: 0.4
- gates_passed: True

## Embeddings contrast

```json
{
  "slice": {
    "chunk_strategy": "fixed_256",
    "top_k": 4,
    "rerank": true,
    "retrieval": true,
    "threshold": null,
    "embeddings": "hash",
    "retriever": "dense"
  },
  "backends": {
    "hash": {
      "recall_at_k": 0.5714285714285714,
      "attribution_rate": 0.4857142857142857,
      "n": 35,
      "hit_vector": [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        0
      ]
    },
    "tfidf": {
      "recall_at_k": 0.4,
      "attribution_rate": 0.37142857142857144,
      "n": 35,
      "hit_vector": [
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0
      ]
    },
    "minilm": {
      "recall_at_k": 0.9428571428571428,
      "attribution_rate": 0.6857142857142857,
      "n": 35,
      "hit_vector": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    }
  },
  "AC4_semantic": "met",
  "contrast_type": "hash_vs_minilm_semantic",
  "paired": {
    "hash_vs": "minilm",
    "n": 35,
    "mean_delta": -0.37142857142857144,
    "se_paired": 0.09245225236621893,
    "ci_95": [
      -0.5526349860663605,
      -0.19022215679078233
    ],
    "discriminative": true,
    "mcnemar_a_only": 1,
    "mcnemar_b_only": 14
  }
}
```
