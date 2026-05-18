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

# Date range (Wave 1: ISO datetime comparison works correctly)
--metadata-filters '{"event_time": {">=": "2026-05-01T00:00:00+00:00", "<=": "2026-05-31T23:59:59+00:00"}}'

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

    # — Wave 1 temporal —
    enable_temporal=False,                # True → parse NL time expressions in query
    temporal_overscan=10,                 # multiply top_k by this when temporal filter present
    temporal_now=None,                    # datetime override for deterministic eval

    # — hybrid + misc —
    gemma=1.0,                            # 1.0 = pure vector, 0.0 = pure BM25
    batch_size=0,
    use_grep=False,
)
```

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

**`enable_temporal=True`** — parse NL time expressions into `event_time` filters. Supported:
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
s.search(query, gemma=0.0)   # 0.0 = pure BM25, 1.0 = pure vector
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
