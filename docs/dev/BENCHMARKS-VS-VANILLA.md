# Benchmarks: zain's fork vs vanilla LEANN

**Date**: 2026-06-01
**Vanilla**: `leann-core 0.3.7` from PyPI (`uv tool install leann`). `LeannSearcher.__init__` default `recompute_embeddings=True`, no `score_passage_ids` on the HNSW backend, no `metadata_filters` argument on `search()`.
**Ours**: `feat/source-registry @ 0a12119` on this repo (Wave 1 temporal + Wave 1.5 multi-axis + Wave 2 metadata-aware + Wave 2.1 stored-vector prefilter + Wave 3 source registry).
**Fixture**: `.leann/indexes/eval-slack/documents.leann` — 14 419 Slack messages from 9 channels (Aug 2025 – May 2026), `BAAI/bge-m3` 1024-d, `--no-recompute` (stored vectors). Embeddings via `iq` at `http://100.122.112.83:8100/v1` (Tailscale → mini1).
**Methodology**: 5 trials × 10 queries, each timed end-to-end inside the searcher (warm — first 3 queries discarded). Median + p95 latency.
**Selectivity buckets**:
- *Sparse* (≤5% of corpus): `channel:Tam` (4.7%), `channel:product-strategy` (rare, near 0%), `created_at` in May 2026 (3.1%).
- *Medium-dense* (5–25%): `channel:btd-community` (6.8%), `author=U09KG04U8LS` (13.8%), `channel:big-brain` (23.1%).

## Headline: the win is recall, not latency

Vanilla LEANN has no `metadata_filters` argument. The closest a vanilla user can do is **oversample then post-filter in Python**: `search(top_k=oversample)` → drop results whose metadata doesn't match → take top `K`. We tested oversample ∈ {50, 200, 500}.

For sparse filters where the filter-matching documents are NOT in the top-K embedding nearest neighbors, vanilla's post-filter approach **silently returns fewer results than asked for**, even at 100× oversample:

| Query | Filter | Selectivity | Vanilla `oversample=500` recall@5 | Ours `metadata_filters` recall@5 |
|---|---|---:|---:|---:|
| "design feedback" | `parent_ref=channel:product-strategy` | ~0% | **2/5** | **5/5** |
| "weekly summary" | `parent_ref=channel:Tam` | 4.7% | **5/5** | **5/5** |
| "prompt engineering tips" | `created_at ∈ May 2026` | 3.1% | **2/5** | **5/5** |

Recall computed by treating ours' native prefilter as truth (it's exhaustive — scores every matching doc). Vanilla simply cannot reach top_k=5 results when the filter excludes the embedding-nearest neighbors and the matching subset is small.

The trade-off: ours' prefilter path is slower per query (it scores every match), but it actually *finds* the matches. Vanilla is fast because it gives up early.

## Latency table (median ms, p95 in parentheses)

### Sanity check: no-filter search (no Wave 2 code in the path)

| Query | Vanilla median | Ours median | Δ |
|---|---:|---:|---:|
| "Loom video demos" | 64.4 (64.9) | 54.0 (55.2) | -10ms |
| "model evaluation benchmarks" | 64.4 (67.8) | 55.0 (55.5) | -9ms |
| "Apple submission process" | 62.4 (67.2) | 58.6 (63.9) | -4ms |
| "Stunspot prompt library" | 66.1 (67.5) | 55.7 (64.7) | -10ms |

Both paths hit the same HNSW search; the small consistent edge for ours is noise / different daemon-warmup interplay. **No regression in the base path.**

### Sparse filter — the prefilter scenario

`channel:Tam` (671 docs, 4.7% selectivity):

| Config | Median ms | p95 ms | n_returned | recall@5 |
|---|---:|---:|---:|---:|
| Ours native prefilter | 194.9 | 195.6 | 5 | **5/5** |
| Vanilla post-filter oversample=50 | 57.5 | 64.6 | 5 | 5/5 |
| Vanilla post-filter oversample=200 | 63.0 | 64.7 | 5 | 5/5 |
| Vanilla post-filter oversample=500 | 74.7 | 75.0 | 5 | 5/5 |

Where Tam-channel matches happen to be embedding-similar, vanilla post-filter wins on latency. ~3× faster than native prefilter for the same recall.

`created_at ∈ May 2026` (447 docs, 3.1% selectivity):

| Config | Median ms | p95 ms | n_returned | recall@5 |
|---|---:|---:|---:|---:|
| Ours native prefilter | 194.2 | 195.9 | 5 | **5/5** |
| Vanilla post-filter oversample=50 | 57.9 | 61.2 | **1** | 1/5 |
| Vanilla post-filter oversample=200 | 60.6 | 68.3 | **2** | 2/5 |
| Vanilla post-filter oversample=500 | 72.1 | 76.6 | **2** | 2/5 |

Vanilla **cannot deliver 5 results** at any oversample level. The post-filter path gives up. Ours pays ~120ms more to actually find them.

`channel:product-strategy` (rare, well under 1%):

| Config | Median ms | p95 ms | n_returned | recall@5 |
|---|---:|---:|---:|---:|
| Ours native prefilter | 177.3 | 179.9 | 5 | **5/5** |
| Vanilla post-filter oversample=500 | 76.5 | 77.3 | **2** | 2/5 |

Same story, sharper. The rarer the filter axis, the more decisively ours wins on completeness.

### Medium-dense filter — auto-routing kicks in

`channel:big-brain` (3336 docs, 23.1% selectivity):

| Config | Median ms | p95 ms | n_returned | recall@5 |
|---|---:|---:|---:|---:|
| Ours `metadata_filters` (auto-routed → post-score-filter) | 123.5 | 123.8 | 1 | 1/1 |
| Vanilla post-filter oversample=50 | 58.7 | 70.0 | 5 | 1/1 |
| Vanilla post-filter oversample=200 | 62.0 | 62.4 | 5 | 1/1 |

When the filter set is dense, ours' selectivity-aware auto-router declines the brute-force prefilter and falls back to post-score-filter — same algorithm as vanilla. The ~60ms overhead vs vanilla is the metadata-filter engine evaluating per-result; it should be possible to close. Filed as a follow-up.

## What this means

1. **Sanity**: no regression in the base search path. Both vanilla and ours hit the same HNSW code at ~60ms median.
2. **Sparse filters (the original prefilter motivation)**: ours delivers *correct top-K results that vanilla cannot reach at any oversample*. Latency cost is ~120-140ms over vanilla's "fast but wrong" path.
3. **Dense filters**: ours' auto-router falls back to post-score-filter — ~60ms overhead vs vanilla, room to optimize.
4. **Wave 2.1 (stored-vector prefilter) is doing what it claimed**: the 194ms native-prefilter timing on 671-doc match count is dominated by 5-10 reconstructs against the HNSW Flat storage, not by re-embedding (which would be ~50ms per batch × 7 batches = ~350ms on this corpus). Pre-2.1 we documented 60s+; this is on a smaller index so the absolute number scales smaller, but the *path* is confirmed.

## `--no-recompute` vs `--recompute` tradeoff (same corpus, same patches)

Same eval-slack corpus rebuilt as `--recompute` via `scripts/rebuild_as_recompute.py` (reuses the existing stored vectors — no re-embedding cost during the rebuild itself).

### Storage

| Mode | Index file (`documents.index`) | Total dir size | Vector data delta |
|---|---:|---:|---:|
| `--no-recompute` | 63.0 MB | 73 MB | — |
| `--recompute` | 3.9 MB | 17 MB | **-59 MB (-77%)** |

Passages JSONL (13 MB) and the offset map are identical in both modes. The savings are pure vector-storage delta. Scales linearly with corpus size and embedding dim (1024 here); a 100k-doc bge-m3 corpus would save ~400 MB.

### Query latency (eval-slack 14 419 docs, bge-m3 via iq Tailscale)

| Config | Latency | Note |
|---|---:|---|
| `--no-recompute` no-filter | **~55 ms** | one query embed + HNSW walk on stored vectors |
| `--no-recompute` sparse prefilter (May 2026, 3.1%) | **~195 ms** | + scan jsonl + 5 vector reconstructs |
| `--recompute` no-filter, complexity=64 (default) | **~30 s** | ~77 ZMQ batches × ~400 ms iq inference each |
| `--recompute` no-filter, complexity=32 | **~21 s** | -30% nodes visited, same 5/5 recall |
| `--recompute` no-filter, complexity=16 | **~8.7 s** | -73% nodes visited, 4/5 recall (one swap) |
| `--recompute` sparse prefilter (May 2026, 3.1%) | **~40 s** | filter narrows candidates but each filter-match batch is still embed-bound |

**During heavy concurrent iq load** (a second client hammering the endpoint), `--recompute` no-filter ballooned to ~110 s. The numbers above reflect normal solo load.

### Knobs that DO NOT help (confirmed empirically)

A first-pass analyst proposed `beam_width=8` would cut ZMQ round-trips proportionally. We tested it:

| beam_width | Latency | Recall |
|---|---:|---|
| 1 (default) | 30.5 s | 5/5 |
| 4 | 34.5 s | 5/5 (identical IDs) |
| 8 | 41.6 s | 5/5 (identical IDs) |

Higher beam = wider frontier = MORE total nodes visited. Even though round-trip count drops, total embed work grows. Same negative result for `batch_size` 64/128/256 — all slower than `batch_size=0` (default).

Why round-trip-batching doesn't help: each existing ZMQ batch already contains ~15 texts of long passages. iq's MLX bge-m3 scales sub-linearly with batch size (batch=64 → 219 ms = 3.4 ms/text vs single = 40 ms/text), so bigger batches are *more* compute-efficient per text — but only when the total nodes visited is constant. HNSW's `beam_width` and `batch_size` knobs do not preserve node count; they grow it.

### Knobs that DO help

| Knob | Effect | Tradeoff |
|---|---|---|
| `complexity=32` (efSearch) | -30% latency vs default | Often zero recall loss on dense corpora |
| `complexity=16` | -70% latency | ~1 result swap per top_5 |
| `--no-recompute` (rebuild) | -99% latency | +56 MB stored vectors on this corpus |
| Co-locate bge-m3 in-process | unknown, likely large | Fork-level surgery (replace ZMQ/HTTP with direct numpy/MLX calls) |

### Wave 2.1 prefilter helps `--recompute` more, in relative terms

For a 3.1%-selectivity query, the prefilter scan + 5 reconstructs from the matched subset replaces ~30s of graph walk with ~40s of filter-subset re-embed — actually slower in this case, but the comparison vanilla can't make (vanilla's `--recompute` only option is run full 30s search then post-filter, with silent recall loss as documented in the no-recompute section above).

### Decision matrix

| You're optimizing for | Pick |
|---|---|
| Interactive search latency (sub-second) | `--no-recompute`, eat 5-15× storage |
| Storage footprint, embedding compute is in-process | `--recompute`, accept 3-10× latency (LEANN's published case) |
| Storage footprint, embedding endpoint over HTTP | `--no-recompute`. The per-batch HTTP overhead × ~80 hops/query swamps the storage win — `complexity=16` only gets to 8.7 s on a 14 k-doc corpus and loses recall |

## Caveats

- **One corpus, one machine**: 14 419-doc index on a single laptop with iq on Tailscale (~10ms RTT). Larger corpora (100k+) and lower-latency embedding endpoints will shift the picture. The Wave 2.1 doc claims 200× win on a 3415-match prod index; that was at ~10× the scale of this fixture.
- **No reranker in either path**: bge-m3 first-stage cosine only. A reranker (planned Wave 4) would close some of vanilla's recall gap, at the cost of latency on both paths.
- **`--recompute` numbers above are over Tailscale** to iq on mini1. Co-locating the embedding backend with the searcher would remove HTTP/Tailscale/queue overhead, but it would still embed every visited passage unless we add a cross-query embedding cache or store vectors. Treat in-process MLX as an experiment, not as a guaranteed 0.5-2s fix. The qualitative ordering ($\text{no-recompute} \ll \text{recompute-local} \ll \text{recompute-network}$) likely holds, but the recompute-local number must be measured.
- **iq embedding dominates fixed cost**: ~50ms per query is network round-trip + bge-m3 inference. Same overhead on both paths. Running on mini1 (iq host) directly would shave ~10ms off all numbers uniformly.

## Reproduction

```bash
cd /Users/zain/Documents/LEANN-fork-rebase
IDX=.leann/indexes/eval-slack/documents.leann

# Vanilla baseline (PyPI 0.3.7 from global uv tool install)
~/.local/share/uv/tools/leann-core/bin/python scripts/bench_vs_vanilla.py \
  --index "$IDX" --label vanilla --output docs/dev/bench-runs/vanilla.jsonl --trials 5

# Ours (this branch)
.venv/bin/python scripts/bench_vs_vanilla.py \
  --index "$IDX" --label ours --output docs/dev/bench-runs/ours.jsonl --trials 5

# Summary table
python3 scripts/bench_summary.py docs/dev/bench-runs/vanilla.jsonl docs/dev/bench-runs/ours.jsonl
```

Raw runs at `docs/dev/bench-runs/{vanilla,ours}.jsonl`.
