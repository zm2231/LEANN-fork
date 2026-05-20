# Experiments

## 2026-05-20: Wave 1.5 Multi-Axis Temporal Eval

Command:

```bash
.venv/bin/python scripts/eval_temporal.py --multi-axis
```

Corpus and gold:

- Wave 1 temporal gold: `tests/eval/temporal_gold.jsonl`, 22 rows.
- Context-layer temporal gold: `tests/eval/context_layer_temporal_gold.jsonl`, 50 rows.
- Context-layer split: 40 non-adversarial rows and 10 adversarial rows.
- Adversarial rows by routed axis: 5 `created_at`, 5 `modified_at`.
- `temporal_now` pinned to `2026-05-19T12:00:00+00:00`.

Summary:

| dataset | mode | rows | precision@5 | recall@5 | mrr | non-adversarial recall@5 |
|---|---|---:|---:|---:|---:|---:|
| temporal_gold | baseline | 22 | 0.11 | 0.36 | 0.31 | 0.36 |
| context_layer_temporal_gold | baseline | 50 | 0.13 | 0.64 | 0.45 | 0.75 |
| temporal_gold | treatment | 22 | 0.25 | 0.77 | 0.59 | 0.77 |
| context_layer_temporal_gold | treatment | 50 | 0.78 | 1.00 | 1.00 | 1.00 |

Acceptance:

- Wave 1 no-regression bar met: treatment R@5 `0.77` >= `0.65`; treatment P@5 `0.25` >= `0.23`.
- Context non-adversarial bar met: treatment non-adversarial R@5 `1.00` >= `0.50`.
- Adversarial bucket was measured but not gated. From the measured context aggregate and non-adversarial recalls, adversarial R@5 was `0.20` in baseline and `1.00` in treatment.

Notes:

- The adversarial rows intentionally use less-direct temporal cues such as `on in May` and `early in May`.
- The treatment path uses multi-axis routing plus hybrid retrieval (`gemma=0.7`) in `scripts/eval_temporal.py`.
- `pkill -f hnsw_embedding_server` after the eval found no leaked process.
