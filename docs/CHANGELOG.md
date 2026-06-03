# Changelog

## Unreleased

### Added

- Adopted upstream PRs after evaluating the open queue against this fork:
  - `leann rebuild <index>` — re-run a build with the index's stored config (incremental delta by default, `--force` for full rebuild). (upstream #326)
  - Optional query log: set `LEANN_QUERY_LOG=<path>` and `LeannSearcher.search()` appends a JSONL record (query, top_k, result ids/scores, query embedding) per call, for offline benchmark replay. (upstream #325)

- Global index manifest for fast `leann list` (`~/.leann/indexes.json`):
  - New `leann.index_manifest` module. Index build (`LeannBuilder.build_index` / `build_index_from_arrays`) upserts an entry; `leann remove` (`_delete_index_directory`) forgets it. `leann list` reads the manifest and only `stat()`-verifies each entry's meta file, instead of `os.walk`-ing every registered project tree in `~/.leann/projects.json`. On a machine with many projects on a slow/external volume, the old walk dominated wall time (e.g. ~34s for 88 indexes across 25 projects); the manifest fast-path replaces thousands of stats with one per index.
  - Entries are keyed by resolved `<name>.meta.json` path and carry what `leann list` renders (name, type, project, size) plus build provenance (backend, embedding model/mode, dimensions). Paths are stored resolved so the current project groups correctly (e.g. `/tmp` vs `/private/tmp`).
  - Self-healing: an empty manifest (fresh/pre-manifest install) falls through to the disk scan and populates it; stale entries (manual `rm -rf`) are pruned on the next `leann list`; `leann list --refresh` rebuilds from a full disk scan as a **non-destructive merge** (keeps still-on-disk entries the scan can't reach — e.g. app indexes under unregistered projects — and overlays freshly scanned size/grouping).
  - Builds on upstream's `fc05933` list-perf rewrite (which still used an unbounded `rglob`) and the fork's `b308c55`/`6867909` `os.walk`+`_SKIP_DIRS` bounding; the manifest removes the per-list walk entirely for the common case.

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
