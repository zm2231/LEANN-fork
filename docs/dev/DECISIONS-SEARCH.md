# Search-time decisions — deep reference

Companion to `.claude/skills/leann-search/SKILL.md`. The skill has the quick tables; this file has the *why* and the failure modes.

## Recompute (search side)

The search flag must match the build flag. Read it back from meta.json:

```bash
jq '.is_recompute, .is_compact, .embedding_model' .leann/indexes/<name>/meta.json
```

| `is_recompute` | Required at search | Failure mode if wrong |
|---|---|---|
| `true` | Embedding server reachable (iq at 100.122.112.83:8100) | Search hangs on ZMQ connect, or returns empty results |
| `false` | Nothing — embeddings are in the index | If you pass `--recompute` anyway, the recomputed vectors will diverge from stored ones — silently wrong scores |

The CLI auto-detects from meta.json; the Python API does too unless you pass `recompute_embeddings=True/False` explicitly. **Don't pass it explicitly unless you know why.**

## `top_k`, `complexity`, `beam_width`

| Param | What it controls | Default | Knob direction |
|---|---|---|---|
| `top_k` | Results returned | 5 | More results = larger candidate pool |
| `complexity` | ANN beam size during traversal | 64 | Higher = more candidates examined = higher recall, slower |
| `beam_width` | (DiskANN) parallel exploration | 1 | Higher = more SSD reads in flight |

Rules of thumb:
- For interactive agent tool calls: `top_k=5, complexity=64`.
- For diversified retrieval where you'll post-rank: `top_k=30, complexity=128`.
- Going past `complexity=128` on HNSW rarely helps; the graph is shallow by design.
- DiskANN benefits more from higher complexity (and `beam_width=4-8` on fast SSDs).

## Hybrid `gemma` weight

`gemma` blends vector and BM25 scores: `final = gemma * vector + (1 - gemma) * bm25`.

| `gemma` | Behavior | When |
|---|---|---|
| 1.0 (default) | Pure vector | Semantic / conceptual queries |
| 0.7 | Vector-leaning hybrid | Mostly semantic but proper nouns / code symbols should count exactly |
| 0.5 | Balanced | Mixed queries (general agent retrieval) |
| 0.3 | BM25-leaning hybrid | Keyword-heavy with semantic safety net |
| 0.0 | Pure BM25 | Exact-string / `grep` replacement |

BM25 doesn't need the embedding server, so `gemma=0.0` is also the fallback if iq is down.

## Metadata filter workflow

This is the most common failure mode for new users: write a filter, get zero results, can't tell why.

### 1. Inspect the corpus first

```python
s = LeannSearcher("my-docs")
print(s.facets(["source_type", "author", "parent_ref"]))
# {"source_type": {"slack": 13062, "document": 14, ...},
#  "author": {"U09KG04U8LS": 1234, ...},
#  "parent_ref": {"channel:btd-community": 5421, ...}}
```

This tells you (a) what values exist, (b) frequency — so you can predict selectivity before searching.

### 2. Write the filter

```python
results = s.search(
    "what did Jay say about preeminence",
    metadata_filters={"source_type": {"==": "slack"},
                      "parent_ref":  {"==": "channel:big-brain"}},
    top_k=10,
)
```

### 3. If zero results — diagnose

```python
results, diag = s.search(query, metadata_filters={...}, explain_filters=True)
```

`diag` is a dict:
```python
{
  "total_passages": 14_000,
  "filter_matches": 0,           # ← problem is here
  "filter_selectivity": 0.0,
  "prefilter_mode_used": "no_filter",
  "ann_candidates_requested": 50,
  "ann_candidates_returned": 50,
  "postfilter_survivors": 0,
  "results_returned": 0,
}
```

Decision tree:
- `filter_matches == 0` → filter is wrong or corpus lacks that value. Re-check with `facets()`.
- `filter_matches > 0, results == 0` → ANN missed them. Set `prefilter="always"`.
- `filter_matches > 0, results > 0, but wrong stuff` → filter is too loose. Add another constraint.

### 4. Diversify if one doc dominates

```python
s.search(query, diversify_by="source_document_id", max_per_group=2)
```

Caps results per group. Critical when one verbose document (a long Slack thread, a multi-page PDF) dominates the top-k.

### 5. Add context if hits look stripped

```python
s.search(query, context_window=1)        # 1 sibling before + 1 after each hit
# or, post-hoc:
hit_ctx = s.expand_context(hit, before=2, after=2)
```

`siblings` is a list on each `SearchResult`. Only fires when `source_document_id` + `chunk_seq` are present in metadata.

## `prefilter` mode

The Wave 2 sparse-filter primitive. Default `"auto"` covers ~all cases; override only when you know better.

### What it does

When `prefilter="auto"`:
1. Estimate selectivity (matching passages / total). Fast — uses metadata index, not embeddings.
2. If the backend is `flat`, or selectivity < `prefilter_threshold` (default 0.05 = 5%) on ANN backends:
   - **Brute-force prefilter**: score every matching passage against the query, return top-k. Bypasses ANN entirely.
   - **On `flat` indexes**: this is the default for vector metadata-filtered searches because flat search is already exhaustive; ANN post-filtering adds false-zero risk without a useful speed trade-off.
   - **On `--no-recompute` indexes (Wave 2.1)**: scores stored FAISS vectors via `score_passage_ids()` instead of re-embedding. ~200× faster on real corpora (307ms vs 60s+ on 3415-match queries against a 9835-chunk index).
   - **On `--recompute` indexes**: falls back to embed-and-score (still correct, but slow). Embedding server must be reachable.
3. Otherwise:
   - **ANN-then-postfilter**: standard path. ANN returns candidates, filter prunes.

### When to override

| Mode | Use when |
|---|---|
| `"auto"` (default) | Always start here |
| `"always"` | Filter matches are tiny *and* you need guaranteed coverage (compliance / audit, single-thread retrieval, "find every mention by user X") |
| `"never"` | Restoring pre-fork behavior for A/B comparison, or debugging |

The `prefilter_threshold` knob (default 0.05) shifts where auto-mode flips on ANN backends. Flat indexes ignore the threshold and use the prefilter path for vector metadata-filtered auto searches.

## Temporal (`enable_temporal=True`)

When on, the searcher parses NL time expressions out of the query and routes them to one of four temporal axes:

| User intent | Routed axis | Examples |
|---|---|---|
| Creation / file birth | `created_at` | "created in May", "files made yesterday" |
| Modification / edit | `modified_at` | "updated last week", "changed on Monday" |
| Real-world event / message / meeting time | `event_time` | "meetings in April", "messages from yesterday" |
| Index ingestion | `indexed_at` | "indexed today", "added to the index this week" |

If the routed axis is missing, production search falls back through adjacent axes so older indexes still return useful results. Set `temporal_strict=True` to disable fallback, or pass `temporal_axis="created_at" | "modified_at" | "event_time" | "indexed_at"` when agent code has already inferred the axis.

### What's parsed

- Relative: `"N {hours,days,weeks,months,years} ago"`, `"last/this {week,month,year,Monday,...}"`, `"yesterday"`, `"today"`, `"tomorrow"`
- Absolute: `"in January"`, `"on March 5th"`, `"since 2026"`
- Ranges: `"between X and Y"`
- Fuzzy: `"around new year"`, `"early March"`, `"first week of January"`, `"around the holidays"`, `"around July 4"`

### Behavior

- The matched time tokens are **stripped** from the semantic query before embedding. "what did Jay say last week" embeds as "what did Jay say".
- `top_k` is overscanned by `temporal_overscan` (default 10×) because temporal filtering is post-hoc on the ANN candidates.
- `temporal_now=datetime(...)` overrides "now" for deterministic eval.
- Caller-supplied temporal `metadata_filters` **win** over parsed values (caller precedence).
- `explain_filters=True` reports `temporal_axis_routed`, `temporal_axis_fallback_used`, `temporal_strict`, and synthesized-axis counts.

### When to turn off

Almost never. The parser is conservative — queries without time tokens are pass-through. Turn off only when:
- Feeding pre-templated queries where time words are literal corpus terms (rare).
- Benchmarking against baseline LEANN behavior.

## ReActAgent — multi-index routing

```python
agent = ReActAgent(searchers={
    "raw": LeannSearcher("raw-sources"),
    "ev":  LeannSearcher("evidence"),
})
agent.run("compare X across raw and evidence", metadata_filters={"source_type": {"==": "slack"}}, enable_temporal=True)
```

Each searcher gets exposed as a named tool (`search_raw`, `search_ev`). The LLM picks which to call.

- Caller-supplied `metadata_filters` are **merged** with LLM-parsed `filter={...}` from tool syntax.
- Caller wins on key conflict.
- `enable_temporal=True` applies to every search the agent issues.

## Diagnostic recipes

### "Why did this query return 0 results?"
```python
results, diag = s.search(query, metadata_filters={...}, explain_filters=True)
print(diag)
```

### "How balanced is this corpus?"
```python
print(s.facets(["source_type", "author"]))
```

### "What's the recompute / model setup for this index?"
```bash
jq '.is_recompute, .is_compact, .embedding_model, .embedding_mode' .leann/indexes/<name>/meta.json
```

### "Inspect routed temporal axis and fallback"
```python
results, diag = s.search(
    "files modified in May",
    enable_temporal=True,
    explain_filters=True,
)
print(diag["temporal_axis_routed"])
print(diag["temporal_axis_fallback_used"])
```

### "Pure-keyword search, no embeddings"
```python
s.search(query, gemma=0.0)
```

### "Cheap retrieval test without iq"
```bash
# Builds with --no-recompute → no embedding server needed at search time
leann build test-idx --docs ./docs --no-recompute --embedding-mode sentence-transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2
```
