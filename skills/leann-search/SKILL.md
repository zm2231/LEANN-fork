---
name: leann-search
description: Search LEANN indexes through CLI, MCP, or Python. Use for `leann search`, `ask`, `react`, metadata filters, hybrid BM25/vector search, temporal queries, sparse-filter diagnostics, result diversification, context windows, facets, and batched multi-search.
---

# LEANN Search

Use this when querying existing LEANN indexes.

## First Decision: Search Mode

| Need | Use |
|---|---|
| Fast top-k retrieval | `leann search <index> "query"` |
| LLM answer over retrieved context | `leann ask <index> "question"` |
| Multi-step or multi-index reasoning | `leann react <index[,index]> "question"` |
| Agent/tool search | MCP `leann_search` or `leann_multi_search` |
| Programmatic control | `LeannSearcher.search()` |

Default to vector search. Use hybrid/BM25 when names, exact phrases, IDs, or proper nouns matter.

| Query type | `vector_weight` |
|---|---|
| Semantic prose | `1.0` |
| Semantic plus exact terms | `0.7` |
| Keyword-heavy with semantic safety net | `0.3` |
| Exact/BM25 only | `0.0` |

## CLI Search

```bash
leann search docs "account proposal" --top-k 10 --json

leann search docs "proposal" \
  --metadata-filters '{"top_folder":{"==":"BD"}}' \
  --vector-weight 0.4 \
  --prefilter auto \
  --explain-filters \
  --diversify-by source_document_id \
  --max-per-group 2 \
  --json
```

Important CLI flags:

- `--top-k N`
- `--complexity N`
- `--vector-weight FLOAT`
- `--metadata-filters JSON`
- `--prefilter auto|always|never`
- `--prefilter-threshold FLOAT`
- `--explain-filters`
- `--diversify-by FIELD`
- `--max-per-group N`
- `--recompute` / `--no-recompute` only for debugging; default auto-detects from `meta.json`
- `--show-metadata`
- `--json`
- `--non-interactive`

CLI does not expose every Python control. Use Python or MCP for `context_window`, `enable_temporal`, `temporal_*`, `query_embedding`, `batch_size`, and `use_grep`.

## Metadata Filters

Format:

```json
{"field_name": {"operator": "value"}}
```

Operators:

- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Membership: `in`, `not_in`
- String: `contains`, `starts_with`, `ends_with`
- Boolean: `is_true`, `is_false`

ISO 8601 datetime strings compare correctly.

Examples:

```bash
--metadata-filters '{"top_folder":{"==":"BD"}}'
--metadata-filters '{"folder_path":{"starts_with":"Reports/2026"}}'
--metadata-filters '{"source_type":{"in":["slack","daily_summary"]}}'
--metadata-filters '{"modified_at":{">=":"2026-05-01T00:00:00+00:00"}}'
--metadata-filters '{"mentioned_urls":{"contains":"loom.com"}}'
```

Generic `build --docs` indexes expose factual filesystem fields such as `top_folder`, `folder_path`, `relative_path`, and `source_root`. Source-specific fields such as `source_type`, `author`, or `event_time` require a source reader or JSONL metadata.

## Sparse Filters And Debugging

Use `prefilter="auto"` by default. It brute-force scores vector metadata-filtered searches on flat indexes, and tiny filtered subsets on ANN backends, to avoid false-zero results.

If a filtered query returns nothing:

```python
results, diag = s.search(
    query,
    metadata_filters={"top_folder": {"==": "BD"}},
    explain_filters=True,
)
print(diag)
```

Interpretation:

- `filter_matches == 0`: filter is wrong or the corpus lacks that field/value.
- `filter_matches > 0` and `results_returned == 0`: on ANN backends, use `prefilter="always"` or raise `prefilter_threshold`.
- One source dominates: use `diversify_by="source_document_id", max_per_group=2`.
- Hits lack context: use `context_window=1` or `expand_context()`.

## Python API

```python
from leann import LeannSearcher

s = LeannSearcher("docs")
results = s.search(
    "account proposal",
    top_k=10,
    vector_weight=0.4,
    metadata_filters={"top_folder": {"==": "BD"}},
    prefilter="auto",
    diversify_by="source_document_id",
    max_per_group=2,
    context_window=1,
)
```

Common Python-only controls:

- `context_window=N`: attach sibling chunks before/after each hit.
- `enable_temporal=True`: parse natural-language time from the query.
- `temporal_axis="created_at|modified_at|event_time|indexed_at"`: override automatic temporal routing.
- `temporal_strict=True`: do not fall back to adjacent time axes.
- `temporal_now=...`: deterministic tests/evals.
- `query_embedding=...`: pass precomputed query vector.
- `use_grep=True`: exact helper path.
- `batch_size=N`: bulk/eval tuning.

`recompute_embeddings=None` means auto-detect from the index metadata. Do not pass `--recompute`/`--no-recompute` unless debugging.

## Temporal Search

Enable temporal parsing when the query mentions time:

```python
s.search("meetings last week about pricing", enable_temporal=True)
```

Routing:

- Created/added/made -> `created_at`
- Modified/updated/changed -> `modified_at`
- Happened/sent/met/meeting -> `event_time`
- Indexed/ingested -> `indexed_at`

Supported expressions include relative windows, absolute dates, ranges, and fuzzy windows such as `last week`, `in January`, `between May and June`, `around new year`, and `early March`.

Temporal parsing strips time tokens from the semantic query and overscans `top_k`.

## MCP Tools

Use MCP as the first-class agent surface. Prefer explicit `index_path` when the
physical index is known; it avoids cwd-dependent name resolution and behaves like
GitNexus' explicit `repo` selector. `index_path` may be either the index directory
(`.../.leann/indexes/name`) or the index prefix (`.../documents.leann`).

Start with:

```text
leann_list()
leann_inspect(index_name="docs")
leann_inspect(index_path="/path/to/.leann/indexes/docs")
```

`leann_list` returns structured index records with `name`, `index_path`,
`project_path`, backend, embedding model, and passage count. Use its `index_path`
directly for follow-up calls.

Then search with the resolved target:

```text
leann_search(
  index_path="/path/to/.leann/indexes/docs",
  query="account proposal",
  top_k=5,
  complexity=32,
)
```

`leann_search` uses the Python API directly, not a shell subprocess:

- Required: `query`, plus one of `index_name` or `index_path`
- Optional: `top_k`, `complexity`, `metadata_filters`, `vector_weight`, `prefilter`, `prefilter_threshold`, `explain_filters`, `include_context`, `diversify_by`, `max_per_group`, `context_window`
- CamelCase aliases are accepted for agent callers, such as `indexPath`, `metadataFilters`, `vectorWeight`, `prefilterThreshold`, `diversifyBy`, `maxPerGroup`, `contextWindow`, `includeContext`, and `explainFilters`

`leann_multi_search` is for batched/paraphrased retrieval:

- `search_mode="prose"` -> `vector_weight=0.3`
- `search_mode="code"` -> `vector_weight=1.0`
- `search_mode="exact"` -> `vector_weight=0.0`
- `search_mode="filtered"` -> larger filtered candidate pool

Required: `query`, plus one of `index_name` or `index_path`.

Optional: `extra_queries`, `search_mode`, `top_k`, `fetch`, `limit`,
`complexity`, `metadata_filters`, `vector_weight`, `prefilter`,
`diversify_by`, `max_per_group`, and `context_window`/`contextWindow`.

`leann_facets` returns metadata value counts before filtered search:

```text
leann_facets(
  index_path="/path/to/.leann/indexes/docs",
  fields=["source_type", "project_id"],
  max_values_per_field=20,
)
```

## Multi Search

```python
result_lists = s.multi_search(
    ["budget discussion", "pricing concern", "contract renewal"],
    top_k=8,
    vector_weight=0.4,
    prefilter="auto",
)
```

`multi_search()` batches query embeddings when semantics are preserved. It falls back to per-query search for recompute indexes, temporal parsing, grep, and pure BM25.

## Ask And React

```bash
leann ask docs "What did the team say about preeminence?"
leann ask docs "What was discussed last week?" \
  --metadata-filters '{"source_type":{"==":"slack"}}'

leann react raw,ev "compare evidence across raw sources and structured eval"
```

`react` supports single-index and multi-index agent workflows.

## Facets And Context

```python
s.facets(["source_type", "author", "top_folder", "folder_path"])

hit = s.search("proposal", top_k=1)[0]
expanded = s.expand_context(hit, before=2, after=2)
```

Use facets before writing filters when you are unsure what metadata exists.

## Query Logging

```bash
LEANN_QUERY_LOG=/tmp/queries.jsonl leann search docs "design feedback"
```

Each search appends JSONL for eval/replay. This is env-gated and requires no code changes.

## Common Pitfalls

- Pure vector search is weak for proper nouns and IDs. Lower `vector_weight`.
- If filtered search misses obvious hits, use `explain_filters=True`, then `prefilter="always"` if matches exist.
- BM25/hybrid requires the BM25 sidecar; current builds create and maintain it by default.
- Folder filters use `top_folder`/`folder_path`; arbitrary business categories require JSONL/source metadata.
- Use `context_window` in Python/MCP, not CLI.
