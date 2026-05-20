# Upgrading to zain/custom-patches (Wave 1 + Wave 2 + fork QoL)

This branch combines, in order on top of upstream `origin/main`:

1. **Wave 1 — temporal substrate** (`feat/temporal-substrate`, 22 commits)
2. **Wave 1.5 — multi-axis temporal** (`feat/multi-axis-temporal`)
3. **Wave 2 — metadata-aware search** (`feat/metadata-aware-search`, 10 commits)
4. **Integration commit** — reconciles Wave 1's `filter_all_passages` with Wave 2's `score_filtered_subset`
5. **Fork's 10 custom commits** — multi-index ReAct, web search via Exa+SearXNG, local-proxy LLM defaults, perf fixes

Final state: branch is **34 commits ahead of `origin/main`**, 99 combined narrow tests pass.

---

## TL;DR — what changed for callers

**Default behavior change (one, opt-out available):**
- `LeannSearcher.search(metadata_filters=...)` with a filter matching <5% of the corpus now uses a **brute-force scored prefilter** instead of ANN-then-postfilter. Returns ranked results from the matching subset instead of false zeros. **Restore old behavior:** `prefilter="never"`.

**Everything else is additive** — new kwargs default to off, new fields are optional, new commands are additions. Existing code keeps working.

**Bug fix (post-Wave-1.5)** — `enable_temporal=True` was returning `top_k * temporal_overscan` (default 10×) results instead of `top_k`. Fixed at commit `3930494`. If you were inadvertently relying on the inflated result count, pass a larger `top_k` explicitly.

---

## What's new — feature index

### Sparse-metadata-filter retrieval that works (Wave 2)

Before: `search("X", metadata_filters={"sai_gate": {"==": "rare_class"}})` could return 0 results even when the filter matched 217 passages in the corpus, because ANN top-k didn't land on any of those 217.

Now: when filter selectivity is below `prefilter_threshold` (default 5%), LEANN brute-force scores the matching passages directly against the query embedding and returns the top-k ranked results from that subset.

```python
results = s.search(
    "rare earth minerals",
    top_k=8,
    metadata_filters={"sai_gate": {"==": "review_gated_machine_transcript"}},
    prefilter="auto",           # default; "always" forces it, "never" disables
    prefilter_threshold=0.05,
)
```

### Filter diagnostics

```python
results, diag = s.search(query, metadata_filters={...}, explain_filters=True)
# diag = {
#   "total_passages": 9835,
#   "filter_matches": 217,
#   "filter_selectivity": 0.022,
#   "prefilter_mode_used": "bruteforce_filtered_subset",
#   "ann_candidates_requested": 5,
#   "ann_candidates_returned": 5,
#   "postfilter_survivors": 0,
#   "results_returned": 8,
# }
```

Use to debug zero-result queries.

### Facet counts

```python
s.facets(["source_type", "author"])
# Single pass over passages.jsonl. Cap each field's distinct-values via max_values_per_field.
```

### Diversify + group cap

```python
s.search(query, diversify_by="source_document_id", max_per_group=2)
# Composite keys also work: diversify_by=["source_document_id", "speaker"]
```

### Context-window sibling expansion

```python
# Inline (every hit gets siblings attached)
s.search(query, context_window=1)
# → SearchResult.siblings: list[SearchResult] | None

# Post-call (explicit, per-hit)
s.expand_context(hit, before=2, after=2)
```

Requires the chunk's metadata to have `source_document_id` and `chunk_seq` (new reserved SIGNALS fields).

### Natural-language time queries (Wave 1)

```python
s.search("commits about Windows last week", enable_temporal=True)
# Internally: parses "last week" → metadata_filters["event_time"] = {">=": ..., "<=": ...}
# Query embedded as just "commits about Windows" (time tokens stripped)
```

Supported expressions:
- `"N {hours,days,weeks,months,years} ago"`
- `"last/this {week,month,year,Monday,...}"`
- `"in January"`, `"on March 5th"`, `"since 2026"`
- `"between X and Y"`
- `"yesterday"`, `"today"`, `"tomorrow"`
- `"around new year"`, `"early March"`, `"first week of January"`
- `"around the holidays"`, `"around July 4"`

`temporal_overscan` multiplies `top_k` (default 10x) when a temporal filter is added, so date-narrow queries still return enough candidates.

### Multi-axis temporal routing (Wave 1.5)

Temporal search now distinguishes four axes instead of collapsing everything into `event_time`:

| Axis | Meaning |
|---|---|
| `created_at` | File/message/document creation time |
| `modified_at` | Last edit/update time |
| `event_time` | Real-world event, meeting, message, or commit author time |
| `indexed_at` | When LEANN ingested the chunk |

`enable_temporal=True` routes natural language to the likely axis: "created in May" uses `created_at`, "modified last week" uses `modified_at`, "meetings yesterday" uses `event_time`, and "indexed today" uses `indexed_at`.

Production search falls back across adjacent axes when older chunks lack the routed field. For exact structured intent, use:

```python
s.search(
    "files modified in May",
    enable_temporal=True,
    temporal_axis="modified_at",
    temporal_strict=True,
)
```

`explain_filters=True` now reports temporal-axis diagnostics, including the routed axis and fallback use.

### Datetime-aware metadata filters

```python
# ISO 8601 strings now compare as datetimes, not strings
metadata_filters={"event_time": {">=": "2026-05-01T00:00:00+00:00", "<=": "2026-05-31T23:59:59+00:00"}}
```

The `<`, `<=`, `>`, `>=` operators detect ISO 8601 per-call and compare timezone-aware. Numeric comparisons still work — no breaking change.

### `indexed_at` stamping (Wave 1)

Every chunk built via `index-*` commands gets `indexed_at` (UTC ISO) stamped at ingest. Filterable like any metadata field. Use to answer "what's new in the index since X".

### Calendar reader fix (Wave 1)

`index-calendar` now emits canonical:
- `event_time` (UTC ISO 8601 with `T`)
- `event_time_local` (with offset)
- `source_type=calendar`
- `source_id=<event_id>`

Old behavior: SQLite-formatted localtime string with space separator that broke `datetime.fromisoformat()` parsing.

### CLI: `--metadata-filters` on `ask`

Upstream PR #316 wired it through. Same JSON format as `leann search --metadata-filters`.

### LeannSearcher project-local index names (fork QoL)

```python
LeannSearcher("my-index")   # resolves to .leann/indexes/my-index/documents.leann
```

Avoids hardcoding absolute paths in app code.

### HNSW + recompute incremental refusal (fork QoL)

```python
LeannBuilder(backend_name="hnsw", is_recompute=True).update_index(...)
# Now raises ValueError before embedding — old behavior silently corrupted passages.jsonl
```

Switch to `--backend-name ivf` for truly incremental, or use `--force` for full rebuild.

### Multi-index ReAct routing (fork)

```python
from leann.react_agent import ReActAgent

agent = ReActAgent(searchers={
    "raw":  LeannSearcher("raw-sources"),
    "ev":   LeannSearcher("evidence"),
})
# Agent prompt exposes search_raw("query"), search_ev("query") as separate tools
```

Plus inline filter passthrough:
```
Action: search_raw("question", filter={"channel": "general"})
```

### ReActAgent.run() — new kwargs

```python
agent.run(question,
    top_k=5,
    metadata_filters={...},     # NEW: merged with LLM-parsed filter= clauses, caller wins
    enable_temporal=True,       # NEW: forward to underlying searcher
)
```

### Web search via Exa + SearXNG (fork)

Drops the Serper dependency. Set `EXA_API_KEY` and/or `SEARXNG_URL` env to enable. Tools `web_search` and `visit_page` exposed to the ReAct agent when configured.

### Local-proxy LLM default (fork)

ReAct LLM defaults to `http://127.0.0.1:8317/v1` with `gpt-5.5` model when no `llm_config` is passed. Override:
- `LEANN_OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `LEANN_LLM_MODEL`

### SIGNALS schema additions

Reserved metadata fields (use these names in your readers):

**Temporal (Wave 1):**
- `event_time` — UTC ISO 8601, when the underlying event happened
- `event_time_local` — local time with offset (optional supplement)
- `indexed_at` — UTC ISO 8601, when LEANN ingested the chunk

**Identity:**
- `author` — single entity responsible (stable ID preferred)
- `activity_type` — `"authored"` | `"received"` | `"visited"` | `"modified"` | `"created"`
- `participant_ids` — list of all participants

**Source:**
- `source_type` — enum (`document`, `git_commit`, `slack`, `daily_summary`, `email`, `imessage`, `browser_history`, `calendar`, `chatgpt`, `claude`, `wechat`, `whatsapp`, `github`, `code`, `voice_memo`, `journal`, `notion`)
- `source_id` — stable per-source identifier
- `source_url` — deep link
- `project_id` — repo/scope
- `parent_ref` — pointer to containing structure (thread, channel, PR)

**Referential (for connectors):**
- `mentioned_urls`, `mentioned_refs`, `mentioned_files` — list of inline references

**Document hierarchy (Wave 2):**
- `source_document_id` — stable doc ID (drives `diversify_by`/`context_window`)
- `chunk_seq` — int sequence within source_document_id (drives sibling adjacency)
- `event_id` — cross-source event identifier (reserved for connector layer)
- `source_url`, `event_date`, `speaker`, `public_citation_allowed`, `review_required` — citation/provenance group

See `docs/dev/SIGNALS.md` for the full table.

---

## Migration recipes

### 1. Existing code keeps working

You don't need to do anything if you're happy with current behavior. All new kwargs are off by default; the one behavior change (sparse-filter prefilter) is a strict improvement on accuracy.

### 2. If you want to keep the OLD sparse-filter behavior

Add `prefilter="never"` to every `.search()` call with metadata_filters. Or rely on the fact that for dense filters (>5% of corpus), behavior is identical.

### 3. To use NL time queries

Switch `s.search(query)` → `s.search(query, enable_temporal=True)` when the query contains time expressions. Or set it globally in your wrapper.

### 4. To diagnose why filter queries return nothing

```python
results, diag = s.search(query, metadata_filters={...}, explain_filters=True)
print(diag)
```

If `filter_matches > 0` but `results_returned == 0`, force `prefilter="always"`.

### 5. To migrate readers to canonical SIGNALS

For each reader, ensure chunks have at minimum:
- `source_type` (the enum string for your source)
- `event_time` (UTC ISO) where applicable
- `author` (stable ID) where applicable

Upstream PR #315 already did `apps/history_data/history.py` (browser_rag) and the broader `chunking_utils.py` metadata propagation, so custom fields now reach the index automatically.

### 6. To use bge-m3 via iq / local infinity

```bash
leann build my-docs --docs ./documents \
  --embedding-mode openai \
  --embedding-model BAAI/bge-m3 \
  --embedding-api-base http://100.122.112.83:8100/v1 \
  --embedding-api-key iq-local
```

### 7. If you were relying on HNSW + recompute incremental rebuild

It now raises `ValueError` instead of silently corrupting. Either:
- Use `--force` for full rebuild, or
- Switch to `--backend-name ivf` for true incremental (add + remove + modify)

### 8. To use multi-index ReAct routing

```python
agent = ReActAgent(searchers={"alias": LeannSearcher("name"), ...})
# Prompt automatically exposes search_alias() tools
```

Replaces single `ReActAgent(searcher=...)` pattern; both still work.

---

## Compatibility matrix

| Caller pattern | After upgrade |
|---|---|
| `s.search(q, top_k=5)` | Unchanged |
| `s.search(q, metadata_filters={...})` dense filter | Unchanged |
| `s.search(q, metadata_filters={...})` sparse filter | **Returns ranked subset results instead of false zero.** `prefilter="never"` restores old. |
| `LeannSearcher(absolute_path)` | Unchanged |
| `LeannSearcher("name")` | **NEW**: resolves to `.leann/indexes/name/documents.leann` |
| `LeannBuilder(...).update_index()` on HNSW+recompute | **Raises ValueError now** (was silent corruption) |
| `ReActAgent(searcher=...)` | Unchanged |
| `ReActAgent(searchers={...})` | NEW (fork) |
| `agent.run(q)` | Unchanged |
| `agent.run(q, metadata_filters={...}, enable_temporal=True)` | NEW (Wave 1) |
| Date-string `metadata_filters` with `<`, `>` ops | **Now correct** (was string-compare only) |
| `leann ask --metadata-filters ...` | Upstream #316 (in baseline) |
| `leann build --use-ast-chunking` | Unchanged, requires `pip install astchunk` in venv |

---

## Verification

After installing this branch:

```bash
# Verify all backends register
python -c "from leann.api import get_registered_backends; print(get_registered_backends())"
# Expect: ['hnsw'] minimum; add 'diskann' and 'ivf' if those backends installed

# Verify temporal module
python -c "from leann.temporal import parse_temporal_query; print(parse_temporal_query('commits last week'))"

# Run narrow regression
.venv/bin/pytest \
  tests/test_metadata_filter_datetime.py \
  tests/test_indexed_at.py \
  tests/test_signals_schema.py \
  tests/test_temporal_parser.py \
  tests/test_search_temporal.py \
  tests/test_react_temporal.py \
  tests/test_react_multi_index.py \
  tests/test_facets.py \
  tests/test_score_filtered_subset.py \
  tests/test_prefilter_integration.py \
  tests/test_explain_filters.py \
  tests/test_diversify.py \
  tests/test_context_window.py \
  -x
# Expect: 99 passed
```

If anything fails, see `docs/dev/PROGRESS.md` and the per-wave plan docs (`TEMPORAL-PLAN.md`, `METADATA-AWARE-PLAN.md`) for the exclusion list for known-pre-existing macOS issues (OpenClaw, DiskANN native SIGSEGV on tiny test fixtures, document_rag subprocess fixture hang).

---

## Roadmap of what's NOT in this branch yet

Documented at the bottom of `TEMPORAL-PLAN.md` and `METADATA-AWARE-PLAN.md`:

- **Wave 3 — reader migrations** (slack, email, imessage, etc. emit full SIGNALS canonical fields)
- **Wave 4 — federated multi-index temporal merge** (`leann search a,b "q" --merge-by event_time`)
- **Wave 5 — temporal decay scoring** (`gemma_temporal` recency weight)
- **Wave 6 — connector layer + sidecar edge store** (referential / participant / hierarchical cross-source edges)
- **leann-memory** — separate repo, Dakera-style importance/decay/TTL/consolidation wrapping LeannSearcher
- **HNSW masked search** — push filter into C++ graph traversal for medium-sparsity perf
