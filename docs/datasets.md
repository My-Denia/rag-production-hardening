# Datasets

See also [DATA_CARD.md](../DATA_CARD.md).

## Corpus

| File | Topic (fictional) |
| --- | --- |
| `data/corpus/eng_oncall.txt` | On-call / engineering ops |
| `data/corpus/facilities_hq.txt` | Facilities |
| `data/corpus/finance_expenses.txt` | Expenses policy |
| `data/corpus/hr_leave_policy.txt` | Leave |
| `data/corpus/hr_remote_work.txt` | Remote work |
| `data/corpus/it_security.txt` | IT security |
| `data/corpus/legal_nda.txt` | NDA / legal |
| `data/corpus/product_helix.txt` | Product Helix |
| `data/corpus/product_nova.txt` | Product Nova |
| `data/corpus/sales_discounting.txt` | Sales discounting |

All text is synthetic.

## Label files

| Path | Notes |
| --- | --- |
| `data/labels_new.jsonl` | Full stratified set from last recompute |
| `data/dev/labels.jsonl` | Dev split |
| `data/holdout/labels.jsonl` | Sealed holdout |
| `data/holdout/manifest.json` | Holdout manifest |
| `data/regression_v1/labels.jsonl` | Frozen 35 |
| `data/regression_v1/manifest.json` | Freeze manifest |
| `data/split_manifest.json` | Split provenance |
| `data/labels.jsonl`, `data/labels_v1.jsonl` | Historical |

## Schema (typical label)

```json
{
  "qid": "n_lex_01",
  "question": "...",
  "answerable": true,
  "category": "lexical",
  "gold_spans": [{"doc_id": "product_nova", "start": 0, "end": 42}],
  "must_contain": ["99.9%"]
}
```

Exact fields may include span encodings used by the metrics module; validate with:

```bash
python -c "from rag_bench.validate_labels import validate; raise SystemExit(validate('data/dev/labels.jsonl'))"
```

(Holdout validate requires unlock env.)

## License

MIT with the repository. No third-party document licenses apply to corpus text (original synthetic).
