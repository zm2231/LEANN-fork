---
name: leann-search
description: Search LEANN indexes (zain's fork — Wave 1 temporal + Wave 2 metadata-aware). Use when the user wants to search/query/retrieve from an index, apply metadata filters, ask natural-language time-window questions ("last week", "around new year"), get explain diagnostics, diversify by document, attach sibling-chunk context, or run a multi-index ReAct agent. Covers `leann search`, `leann ask`, `leann react`, and `LeannSearcher.search()`.
---

# Searching LEANN indexes (zain's fork)

Three CLI entry points + Python API. Embeddings flow through iq at `http://100.122.112.83:8100/v1` (Tailscale, works from any device).

| What you want | Use |
|---|---|
| One-shot vector search, top-k results | `leann search` |
| LLM-answered question against the index | `leann ask` |
| Multi-turn ReAct agent (single or multi-index) | `leann react` |
| Programmatic from Python | `LeannSearcher.search()` |

## Capabilities at a glance

All opt-in, all composable in a single `search()` call:

- **Vector search** — dense ANN over bge-m3 embeddings (default).
- **Hybrid search** — `vector_weight` blends vector + BM25/FTS5 keyword (1.0 = pure vector, 0.0 = pure keyword). Scores are **min-max normalized per source** before fusing, so the weight is an honest linear blend (a raw blend was BM25-dominated at every weight — fixed 2026-06). L2 indexes are negated first; cosine/IP unaffected.
- **Metadata filters** — `metadata_filters={"field": {"op": value}}`. Operators: `==, !=, <, <=, >, >=, in, not_in, contains, starts_with, ends_with, is_true, is_false` (datetime-aware on ISO 8601). Multiple fields = AND; `{">=": a, "<": b}` on one field = range. Full reference below.
- **Sparse-filter prefilter** — `prefilter="auto"` brute-force-scores tiny filtered subsets (selectivity <5%) to avoid ANN false-zeros.
- **NL temporal** — `enable_temporal=True` parses natural-language time straight from the query into axis-routed filters: relative (`"3 weeks ago"`, `"last month"`, `"yesterday"`), absolute (`"in January"`, `"on March 5th"`, `"since 2026"`), ranges (`"between X and Y"`), and fuzzy (`"around new year"`, `"early March"`, `"first week of January"`, `"around the holidays"`). It **auto-routes to the right axis by verb** — *created/added* → `created_at`, *modified/updated* → `modified_at`, *happened/sent/met* → `event_time`, *indexed/ingested* → `indexed_at` — then **strips the time tokens from the semantic query** (so the embedding stays clean) and overscans `top_k`. `temporal_axis=` overrides routing; `temporal_strict=True` disables fall-through to adjacent axes; `temporal_now=` pins "now" for deterministic eval.
- **Diversify** — `diversify_by="source_document_id", max_per_group=N` caps results per group.
- **Context window** — `context_window=N` attaches N sibling chunks before+after each hit.
- **Facets** — `s.facets([...])` returns corpus value counts for filter authoring.
- **Explain** — `explain_filters=True` returns `(results, diagnostics)` with selectivity + routing.
- **Query log** — `LEANN_QUERY_LOG=<path>` appends a JSONL record per search for replay.
- **Bulk search** — `s.multi_search([q1, q2, …], top_k=…)` embeds all queries in ONE backend call, then runs each through the normal `search()` path with its precomputed embedding. For `--no-recompute` bulk/eval workloads this pays the per-query embedding round-trip once instead of N times (~4–5× warm on 8–10 queries; batched embeddings are numerically identical to single-query). Accepts every `search()` kwarg. Transparently falls back to per-query `search()` when batching can't help or wouldn't match — recompute mode, `enable_temporal=True` (strips time tokens before embedding), `use_grep`, or pure-BM25 (`vector_weight=0.0`) — so it's always safe to call.

## `leann search` — fast retrieval

```bash
leann search my-docs "what is RAG?"
leann search my-docs "preeminence" --top-k 10 --json
leann search my-docs "fix" --show-metadata

# Metadata filter
leann search my-docs "Slack BTD" \
  --metadata-filters '{"parent_ref": {"==": "channel:btd-community"}}'
```

### Key flags
- `--top-k N` — default 5
- `--complexity N` — search complexity (default 64)
- `--recompute` / `--no-recompute` — must match how index was built
- `--json` — machine-readable
- `--non-interactive` — skip prompts
- `--show-metadata` — display source paths / metadata
- `--metadata-filters JSON` — filter (see below)
- `--daemon` / `--no-daemon` — ZMQ embedding daemon reuse (default on)
- `--daemon-ttl N` — daemon idle TTL seconds
- `--embedding-prompt-template "query: "` — for asymmetric models

### Metadata filter JSON

Format: `{"field_name": {"operator": value}, ...}`

Operators:
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=` (datetime-aware on ISO 8601 — Wave 1)
- Membership: `in`, `not_in` (value must be a list)
- String: `contains`, `starts_with`, `ends_with`
- Boolean: `is_true`, `is_false`

Examples:
```bash
# Equality
--metadata-filters '{"source_type": {"==": "slack"}, "parent_ref": {"==": "channel:big-brain"}}'

# Date range (Wave 1/1.5: ISO datetime comparison works correctly across temporal axes)
--metadata-filters '{"event_time": {">=": "2026-05-01T00:00:00+00:00", "<=": "2026-05-31T23:59:59+00:00"}}'
--metadata-filters '{"modified_at": {">=": "2026-05-01T00:00:00+00:00"}}'

# Membership
--metadata-filters '{"source_type": {"in": ["slack", "daily_summary"]}}'

# URL substring
--metadata-filters '{"mentioned_urls": {"contains": "loom.com"}}'
```

## `leann ask` — LLM-answered question

```bash
leann ask my-docs "What did Jay say about preeminence?"
leann ask my-docs "What was discussed last week?" \
  --metadata-filters '{"source_type": {"==": "slack"}}'
```

Same `--metadata-filters` syntax as `search` (upstream #316).

## `leann react` — multi-turn agent

```bash
# Single-index
leann react my-docs "complex multi-step question"

# Multi-index (fork — exposes search_raw, search_ev as separate tools)
leann react raw,ev "compare evidence across raw sources and structured eval"
```

The agent natively understands NL time expressions — *"What happened in Slack last week?"* parses `last week` into an `event_time` filter automatically (Wave 1).

## Python API — `LeannSearcher.search()`

```python
from leann import LeannSearcher

# Bare name resolves to .leann/indexes/<name>/documents.leann (fork QoL)
s = LeannSearcher("my-docs")
results = s.search("query", top_k=5)
```

### Full signature (everything new is opt-in)

```python
s.search(
    query,
    top_k=5,

    # — backend tuning —
    complexity=64,
    beam_width=1,
    prune_ratio=0.0,
    recompute_embeddings=None,
    pruning_strategy="global",

    # — metadata filtering —
    metadata_filters=None,                # dict: {"field": {"op": value}}

    # — Wave 2 sparse-filter primitives —
    prefilter="auto",                     # "auto" | "always" | "never"
    prefilter_threshold=0.05,             # selectivity below which prefilter activates
    explain_filters=False,                # True → returns (results, diagnostics_dict)

    # — Wave 2 result shaping —
    diversify_by=None,                    # str | list[str] | None — cap results per group
    max_per_group=2,
    context_window=0,                     # ≥1 → attach N siblings before+after each hit

    # — Wave 1/1.5 temporal —
    enable_temporal=False,                # True → parse NL time expressions in query
    temporal_overscan=10,                 # multiply top_k by this when temporal filter present
    temporal_now=None,                    # datetime override for deterministic eval
    temporal_strict=False,                # True → do not fall back to adjacent temporal axes
    temporal_axis=None,                   # override routed axis: created_at | modified_at | event_time | indexed_at

    # — hybrid + misc —
    vector_weight=1.0,                    # 1.0 = pure vector, 0.0 = pure BM25 (`gemma=` is a deprecated alias)
    batch_size=0,
    use_grep=False,
)
```

### Decision guide

Deeper reference: `docs/dev/DECISIONS-SEARCH.md`.

#### `--recompute` vs `--no-recompute` (search side)

**Auto-detected from `meta.json` — don't pass `recompute_embeddings=` unless you know why.**

```python
LeannSearcher("my-index")                       # auto-detects from meta.json (default)
LeannSearcher("my-index", recompute_embeddings=False)  # explicit override
```

Build-side flag must match how the index was built. Read it back if curious:
```bash
jq '.backend_kwargs.is_recompute' .leann/indexes/<name>/documents.leann.meta.json
```
- Built with `--recompute` → embedding server (iq) must be reachable at search time.
- Built with `--no-recompute` → embeddings already in the index; no server needed at search time. **Plus**: metadata-prefilter queries use the fast stored-vector path (Wave 2.1) instead of re-embedding matches.

#### Picking `top_k` + `complexity`

| Goal | `top_k` | `complexity` |
|---|---|---|
| Quick lookup / agent tool call | 5 | 64 (default) |
| Diversified retrieval before re-rank | 20-50 | 128 |
| Exhaustive (rare — prefer brute-force via `prefilter="always"`) | 100+ | 256 |

`complexity` is the ANN beam — larger = more candidates examined = slower but higher recall. Diminishing returns past 128 on HNSW.

#### Hybrid weight (`vector_weight`)

`vector_weight=1.0` (default) = pure vector. `vector_weight=0.0` = pure BM25. (The old `gemma=` kwarg still works as a deprecated alias but emits a `DeprecationWarning` — use `vector_weight=`.) BM25 is now FTS5-backed (SQLite virtual table, built once at index-build time or on-demand, memory-bounded — upstream sync). Mix:
- **1.0** — semantic queries ("what did Jay say about X")
- **0.7** — mostly semantic but you want exact terms to count (proper nouns, code symbols)
- **0.3** — keyword-heavy queries with semantic safety net
- **0.0** — exact-string / grep replacement

#### Authoring a metadata filter — workflow

1. **Inspect first** — `s.facets(["source_type", "author", "parent_ref"])` to see what values exist.
2. **Write the filter** — `{"field": {"op": value}}`.
3. **If zero results** — re-run with `explain_filters=True`, check `filter_matches`:
   - `filter_matches == 0` → filter is wrong or corpus lacks that value.
   - `filter_matches > 0` but `results == 0` → ANN missed them. Set `prefilter="always"`.
4. **If one doc dominates** — add `diversify_by="source_document_id", max_per_group=2`.
5. **If hits lack context** — add `context_window=1` (or use `s.expand_context(hit)` post-hoc).

#### `prefilter` mode — when to override the default

| Mode | When |
|---|---|
| `"auto"` (default) | Always start here. Routes to brute-force when selectivity <5%, ANN-postfilter otherwise. |
| `"always"` | Filter matches are tiny *and* you want guaranteed coverage (compliance, single-thread retrieval). |
| `"never"` | Restore pre-fork behavior (debugging, comparing to baseline LEANN). |

#### Temporal — when `enable_temporal=True`

Turn on if the query mentions time. The parser is permissive — leaving it on by default is fine; non-temporal queries are unaffected. Turn off only if you're feeding pre-templated queries where time tokens are literal (rare).

### When to use which kwarg

**`prefilter="auto"` (default)** — **fixes sparse-filter false-zero**. If your filter matches <5% of the corpus, the searcher brute-force scores just the matching passages instead of ANN-then-postfilter. To restore pre-fork behavior: `prefilter="never"`.

**`explain_filters=True`** — returns `(results, diagnostics)`:
```python
{
  "total_passages": int,
  "filter_matches": int,
  "filter_selectivity": float,
  "prefilter_mode_used": "bruteforce_filtered_subset" | "ann_postfilter" | "no_filter",
  "ann_candidates_requested": int,
  "ann_candidates_returned": int,
  "postfilter_survivors": int,
  "results_returned": int,
}
```

Use to debug why a query returns nothing.

**`diversify_by="source_document_id", max_per_group=2`** — cap per-doc results. Critical when one document dominates the top-k.

**`context_window=1`** — attach 1 sibling chunk before + 1 after each hit (by `source_document_id` + `chunk_seq`). Returned as `SearchResult.siblings: list[SearchResult] | None`.

**`enable_temporal=True`** — parse NL time expressions and route them to a temporal axis:
- Creation language (`created`, `added`, `made`) → `created_at`
- Edit/update language (`modified`, `changed`, `updated`) → `modified_at`
- Event/message/meeting language (`happened`, `sent`, `meeting`) → `event_time`
- Index-ingest language (`indexed`, `ingested`, `added to index`) → `indexed_at`

Production search falls back to adjacent axes when the routed field is missing; set `temporal_strict=True` when structured agent intent requires the exact axis. Use `temporal_axis="modified_at"` etc. to bypass NL routing.

Supported:
- `"N {hours,days,weeks,months,years} ago"`
- `"last/this {week,month,year,Monday,...}"`
- `"in January"`, `"on March 5th"`, `"since 2026"`
- `"between X and Y"`, `"yesterday"`, `"today"`, `"tomorrow"`
- `"around new year"`, `"early March"`, `"first week of January"`
- `"around the holidays"`, `"around July 4"`

Strips time tokens from semantic query, auto-overscans `top_k` 10x.

### Facets — corpus introspection

```python
s.facets(["source_type", "author"])
# → {"source_type": {"slack": 13062, "document": 14, ...},
#    "author":      {"U09KG04U8LS": 1234, ...}}
```

Use to understand corpus balance before crafting a filter, or for diagnostic UIs.

### Post-call context expansion

```python
hit = results[0]
hit_with_ctx = s.expand_context(hit, before=2, after=2)
# hit_with_ctx.siblings → list[SearchResult]
```

### Query logging (eval / replay)

Set `LEANN_QUERY_LOG=<path>` in the environment and every `search()` appends one JSONL record (`ts`, `query`, `top_k`, result `id`/`score`, and the query `embedding` when one was computed). Useful for offline benchmark replay and capturing a real query stream. No code change needed — it's purely env-gated (upstream #325).

```bash
LEANN_QUERY_LOG=/tmp/queries.jsonl leann search my-docs "design feedback"
```

## ReActAgent (Python)

```python
from leann import LeannSearcher
from leann.react_agent import ReActAgent

# Single-index
agent = ReActAgent(searcher=LeannSearcher("my-docs"))

# Multi-index (fork — exposes search_raw, search_ev as named tools)
agent = ReActAgent(searchers={
    "raw": LeannSearcher("raw-sources"),
    "ev":  LeannSearcher("evidence"),
})

answer = agent.run(
    "What did Tam say about task triage in January?",
    top_k=5,
    metadata_filters={"source_type": {"==": "slack"}},   # merged with LLM-parsed filter=
    enable_temporal=True,                                # parse NL time on every search
)
```

Caller-supplied `metadata_filters` are merged with any LLM-parsed `filter={...}` from tool syntax; **caller wins on key conflict**.

## Common patterns

### "Why did this query return 0 results?"
```python
results, diag = s.search(query, metadata_filters={...}, explain_filters=True)
print(diag)
# filter_matches == 0 → filter wrong / corpus doesn't have it
# filter_matches > 0 but results == 0 → set prefilter="always"
```

### "Search only in last month, diverse documents, with context"
```python
s.search(
    query,
    top_k=10,
    enable_temporal=True,
    diversify_by="source_document_id",
    max_per_group=2,
    context_window=1,
)
```

### "What channels exist in this slack index?"
```python
print(s.facets(["parent_ref"]))
```

### "Pure-keyword (no embeddings) search"
```python
s.search(query, vector_weight=0.0)   # 0.0 = pure BM25, 1.0 = pure vector
```

## When the user asks

- *"search the index"* / *"find X in Y"* → `leann search Y "X"` or `s.search("X")`
- *"ask the index"* → `leann ask`
- *"agent" / "multi-step" / "multi-index"* → `leann react`
- *"filter by ..."* → `--metadata-filters` or `metadata_filters=`
- *"only from last week"* / *"around X"* / *"this month"* → `enable_temporal=True`
- *"too many hits from one document"* → `diversify_by="source_document_id"`
- *"need surrounding context"* → `context_window=N` or `s.expand_context(hit)`
- *"why no results"* → `explain_filters=True`
- *"what's in this index"* → `s.facets([...])` or `leann list`
- *"date range filter"* → ISO-string `<`/`<=`/`>`/`>=` in metadata_filters (datetime-aware, Wave 1)
