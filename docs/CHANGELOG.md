# Changelog

## Unreleased

### Added

- Wave 1.5 multi-axis temporal search:
  - SIGNALS now defines `created_at`, `modified_at`, `event_time`, `indexed_at`, and temporal-axis diagnostics.
  - Filesystem and specialized readers emit available temporal axes while omitting meaningless `event_time` values.
  - Natural-language temporal search routes to the appropriate axis, supports `temporal_axis` overrides, and supports `temporal_strict=True` for no-fallback behavior.
  - `scripts/build_context_layer_temporal.py` builds a context-layer temporal eval corpus and `scripts/eval_temporal.py --multi-axis` evaluates Wave 1 and Wave 1.5 gold sets together.

### Changed

- Synced fork with upstream `yichuan-w/LEANN` main (12 commits), bringing in:
  - **BM25 FTS5 migration** (upstream #328, #332–335, #341): replaces the in-memory `BM25Scorer` (O(corpus) RAM, fit-on-first-search full scan) with `Fts5BM25Index`, a SQLite FTS5 virtual table built once at index-build time and queried memory-bounded via `bm25()`. Default BM25 backend is now `fts5`. Only affects hybrid/keyword search; the dense + metadata-prefilter product path is unchanged.
  - **MPS memory pathology fix** (upstream #340): drops `torch.mps.set_per_process_memory_fraction(0.9)`, guards `torch.compile(reduce-overhead)` to CUDA only, and calls `torch.mps.empty_cache()` per batch. Footprint on 32 GB Apple Silicon `sentence-transformers` builds drops ~22 GB → ~3 GB. Does not affect the MLX/iq HTTP embedding path.
  - **`gemma` → `vector_weight`** hybrid-weight rename (upstream #324), with a deprecated `gemma=` alias on `search()`/`ask()`.
  - Conflict resolution preserved all Wave 1.5 temporal and Wave 2 prefilter/diversify logic; restored `from collections import Counter, defaultdict` that upstream dropped with `BM25Scorer` but our `_diversify_results`/selectivity code still requires.
