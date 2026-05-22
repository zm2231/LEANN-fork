# Handover — LEANN fork updates since you last picked it up

You're on `feat/multi-axis-temporal` (HEAD `3930494`). This is the fork checkout at `/Volumes/4/GitHub/pi-ult/references/LEANN` on mini1. The global `leann` tool + the spark/cadence-pipeline venvs all point at this checkout via editable install — `git pull` here is enough to pick up changes; no `pip install` needed.

## What's new since Wave 2 (your starting baseline)

You already have Wave 2 (sparse-prefilter, facets, explain_filters, diversify_by, context_window). Layered on top now:

### Wave 1 — Temporal substrate (already shipped before you started)

- `enable_temporal=True` on `LeannSearcher.search()` parses NL time expressions ("last week", "around new year", "in January", "between X and Y") and applies a date filter.
- Datetime-aware metadata filter comparisons (`<`, `<=`, `>`, `>=`) on ISO 8601 strings — auto-detected per call.
- Every chunk auto-stamped with `indexed_at` (UTC ISO) at ingest.

### Wave 1.5 — Multi-axis temporal (new)

The single `event_time` axis split into four. Filters and NL routing now distinguish:

| Axis | Meaning | NL triggers |
|---|---|---|
| `created_at` | When the underlying thing was authored | "created", "authored", "sent", "wrote" |
| `modified_at` | Last source-level change | "edited", "modified", "updated", "changed" |
| `event_time` | Semantic moment of relevance | "happening", "scheduled", "during", bare time tokens |
| `indexed_at` | When LEANN ingested | "indexed", "added to the index" |

Filesystem ingest now stamps `created_at` (from `st_birthtime`) + `modified_at` (from `st_mtime`). Specialized readers backfill per source (browser → first_visit/last_visited, imessage → sent/edited, etc.).

**Production default**: fallback chain — if a chunk lacks the routed axis, search falls back to adjacent axes so mixed-vintage indexes still return results.

**Strict mode**: pass `temporal_strict=True` to opt out of fallback. Use when your tool call already knows the axis from structured intent. Strict-mode chunks lacking the routed axis are excluded.

**Explicit override**: `temporal_axis="created_at" | "modified_at" | "event_time" | "indexed_at"` bypasses NL routing entirely.

**Diagnostics**: `explain_filters=True` now reports `temporal_axis_routed`, `temporal_axis_fallback_used` (list of `(chunk_id, axis_actually_used)`), `temporal_strict`, and `temporal_synthesized_axes` (count of chunks whose routed axis was source-degenerate, e.g. calendar `created_at` synthesized from `event_time`).

### Bug fix (post-Wave-1.5) — `top_k` discipline

`enable_temporal=True` was returning `top_k × temporal_overscan` results instead of `top_k`. Overscan (default 10×) inflates the ANN candidate pool so date filters don't drain it; the inflated value was leaking past the caller's requested cap.

Fix at `3930494`: snapshots requested top_k before overscan and trims final results before return. If your code inadvertently relied on the inflated count, pass a larger explicit `top_k`.

### Wave 2.1 — stored-vector prefilter on `--no-recompute` indexes

Your `score_passage_ids` patch landed (with regression tests + a couple of robustness improvements). On `rubio-raw-sources-bge-metadata-norecompute` with `source_family=official_state_gov_text` (3415 matches), filtered search now runs **~307ms** instead of 60s+. Commit `5cccadd`.

### Follow-up — `recompute_embeddings` auto-detect

The Wave 2.1 patch exposed a footgun: `LeannSearcher` defaulted `recompute_embeddings=True`, so callers had to remember `LeannSearcher(path, recompute_embeddings=False)` to hit the stored-vector fast path. **No longer required.** As of `d87890a`, the default is `None` → auto-read from `meta.json["backend_kwargs"]["is_recompute"]`. Your existing query runner that passes `False` explicitly still works (explicit wins). You can drop the kwarg in new code.

## What this means for your Rubio analysis

You were unblocked by Wave 2's prefilter+facets+explain. Wave 1 + 1.5 give you additional axes for time-bounded queries that should be useful:

```python
# "What did Rubio say in committee hearings in April?"
results, diag = s.search(
    "committee testimony",
    enable_temporal=True,                 # parses "in April"
    metadata_filters={"author": {"==": "rubio"}, "source_type": {"==": "hearing"}},
    explain_filters=True,
    top_k=20,
    diversify_by="source_document_id",    # avoid one verbose hearing dominating
    max_per_group=3,
)

# Strict-axis variant when you know the intent
results = s.search(
    "press releases edited last month",
    enable_temporal=True,
    temporal_axis="modified_at",
    temporal_strict=True,
    top_k=10,
)
```

If your existing corpus was indexed pre-Wave-1.5 (only has `event_time`, no `created_at` / `modified_at`), production fallback keeps your existing queries working. You only need to reindex if you want `created_at`/`modified_at` filterability — Wave 1.5 readers stamp the new axes on fresh ingest.

## Skills + reference docs

- `~/.claude/skills/leann-index/SKILL.md` and `~/.claude/skills/leann-search/SKILL.md` — updated. Both have a "Decision guide" section covering backend choice, recompute tradeoff, chunk sizing, AST chunking, hybrid `gemma` weight, prefilter modes, temporal axis routing, filter authoring workflow.
- `docs/dev/DECISIONS-INDEX.md` and `docs/dev/DECISIONS-SEARCH.md` — deeper reference (cost models, failure modes, `jq` recipes for `meta.json` introspection).
- `docs/UPGRADE.md` — feature-by-feature index + 8 migration recipes + compatibility matrix.

## Things to know

1. **Editable install** — your `leann` command and `from leann import ...` both resolve through `/Volumes/4/GitHub/pi-ult/references/LEANN/packages/leann-core/src/leann/`. To pull updates: `cd` there and `git pull`. No reinstall step.
2. **Embeddings** route through iq at `http://100.122.112.83:8100/v1` (Tailscale). Model: `BAAI/bge-m3`. Check `curl -sf http://100.122.112.83:8100/v1/models` if a search hangs.
3. **DiskANN backend** has a pre-existing native SIGSEGV on test fixtures with <256 vectors. Don't touch DiskANN code paths. If you need that backend, build with ≥256 chunks.
4. **Branch hygiene** — `feat/multi-axis-temporal` is the integration branch. New work should branch from it (or wait for merge to main, TBD).

## Next on this fork (FYI)

- **Wave 3 — Source registry** (planned, not started). Declarative manifest-driven source layer; current 7 hardcoded readers (browser, email, calendar, imessage, wechat, chatgpt, claude) migrate to manifests. Drafted at `docs/dev/SOURCE-REGISTRY-PLAN.md`. Reference architecture: [printing-press-library](https://github.com/mvanhorn/printing-press-library).
- Possible future: reranking (cross-encoder like Jina), GCal reader (would be first Wave 3 manifest consumer).

Questions: ping the upstream session. Source of truth: this checkout + `docs/dev/`.
