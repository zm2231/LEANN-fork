# Wave 1.5 — Multi-axis temporal

Wave 1 collapsed four distinct temporal axes into one (`event_time`). That works for messages but loses information for everything else, and gold-set eval was thin (22 self-authored queries). Wave 1.5 fixes both: split the axes, route NL → axis, backfill readers, and re-evaluate against an agent-style gold set on the context-layer corpus.

This is a Wave 1 patch, not a new wave. Same branch lineage. Closes before Wave 3 starts because the source registry manifests need the axis vocabulary to map correctly.

## Glossary

| Term | Meaning |
|---|---|
| **Axis** | A distinct temporal field on a chunk. There are four: `created_at`, `modified_at`, `event_time`, `indexed_at`. |
| **Routing** | The NL parser deciding which axis a query targets (verb-driven). |
| **Fallback chain** | When a chunk lacks the routed axis, which axes get checked instead. |
| **Strict mode** | Opt-out of fallback (`temporal_strict=True`) — query returns 0 if the routed axis is missing. |
| **context-layer-temporal** | New eval index, rebuilt from `/Volumes/4/GitHub/context-layer` with the multi-axis schema. ~1300 chunks. |

## SIGNALS schema additions

| Field | Type | Meaning | Required when |
|---|---|---|---|
| `created_at` | UTC ISO 8601 | When the underlying thing was authored | Always, if knowable |
| `modified_at` | UTC ISO 8601 | Last change to the underlying thing | If distinct from `created_at` |
| `event_time` | UTC ISO 8601 | Semantic moment of relevance — defaults to `created_at` for messages; differs for calendar (start time), scheduled tasks, dated docs | If different from `created_at` |
| `indexed_at` | UTC ISO 8601 | Auto-stamped at ingest | Auto (Wave 1) |

Invariants:
- `created_at <= modified_at` (when both present)
- `event_time` is independent — can be future-dated (calendar) or past (historical references)
- All fields are UTC ISO 8601 with timezone offset

`docs/dev/SIGNALS.md` gets updated; existing single-`event_time` data stays valid (fallback chain reads it).

## Per-reader backfill matrix

| Reader | `created_at` | `modified_at` | `event_time` |
|---|---|---|---|
| `build` (filesystem) | file ctime | file mtime | (omit) → falls back to `created_at` |
| `index-calendar` | event.created (if iCloud has it) | event.last_modified | event.start_time |
| `index-email` | email.Date header | email.X-Last-Modified (or omit) | Date header |
| `index-imessage` | msg.date | msg.date_edited (iOS 16+) | msg.date |
| `index-browser` | first_visit_time | last_visited | last_visited |
| `index-wechat` | msg.CreateTime | (omit) | CreateTime |
| `index-chatgpt` | conversation.create_time | conversation.update_time | create_time |
| `index-claude` | conversation.created_at | conversation.updated_at | created_at |
| `index-slack` (eval) | ts | edited.ts (if edits[] present) | ts |
| `index-git_commit` (eval) | author_date | commit_date | author_date |

Wave 1 stamped `event_time` only — sometimes from start (calendar), sometimes from sent (messages). Wave 1.5 keeps Wave 1 behavior for `event_time` on those readers and adds the other two axes alongside.

## Parser routing

Verb-driven. Parser scans for axis-indicating tokens before time-window parsing.

| NL token | Routed axis |
|---|---|
| "created", "authored", "sent", "wrote", "posted", "made", "drafted" | `created_at` |
| "edited", "modified", "updated", "changed", "amended", "revised", "touched" | `modified_at` |
| "happening", "scheduled", "during", "starting", "ending", "occurred", "on (date)" for events | `event_time` |
| "indexed", "added to the index", "ingested" | `indexed_at` |
| (no axis verb) | `event_time` with fallback chain |

Examples:
- *"files I created last week"* → `created_at`, window = last week
- *"docs edited yesterday"* → `modified_at`, window = yesterday
- *"meetings happening tomorrow"* → `event_time`, window = tomorrow
- *"what was added to the index today"* → `indexed_at`, window = today
- *"what did Jay say last week"* → `event_time` (no verb), fallback to `created_at`

## Fallback chain

When the routed axis is missing on a chunk:

| Routed | Fallback order |
|---|---|
| `event_time` | `event_time` → `created_at` → `modified_at` |
| `created_at` | `created_at` → `event_time` → `modified_at` |
| `modified_at` | `modified_at` → `created_at` → `event_time` |
| `indexed_at` | `indexed_at` only (no fallback — too distinct from authoring time) |

Opt-out: `s.search(query, enable_temporal=True, temporal_strict=True)` → no fallback, chunk excluded if routed axis absent.

Rationale: in mixed corpora (some chunks pre-Wave-1.5, some post), strict mode is the right default for new eval but the wrong default for production queries. Production gets fallback, eval can flip it off.

## Searcher integration

`LeannSearcher.search()` adds:
```python
s.search(
    query,
    enable_temporal=True,
    temporal_strict=False,   # NEW — default False
    temporal_axis=None,      # NEW — override parser routing ("created_at" | "modified_at" | "event_time" | "indexed_at")
)
```

`temporal_axis=` is an explicit override for agent code that wants to bypass NL routing. Useful when the agent already knows the intent from a structured task spec.

Returned filter object now carries `axis` field. `explain_filters=True` diagnostics include:
- `temporal_axis_routed` — which axis the parser/override picked
- `temporal_axis_fallback_used` — list of (chunk_id, axis_actually_used) when chain fired
- `temporal_strict` — boolean, surfaces the mode used (so regressions don't get misread)
- `temporal_synthesized_axes` — count of chunks whose routed axis was marked synthesized

Chunks where an axis is degenerate (e.g. Apple Calendar `created_at` = `event_time` because source doesn't separate them) carry `<axis>_synthesized: true` in metadata. Backfills emit this where applicable.

## Eval — context-layer agent-style gold

Build `context-layer-temporal` from `/Volumes/4/GitHub/context-layer/` with a small custom reader that stamps file ctime/mtime as `created_at`/`modified_at`. ~1300 chunks. Embedding: `BAAI/bge-m3` via iq.

Gold set: 50 queries split into 5 buckets of 10:
1. **Pure `created_at`** — "docs I wrote last month", "files created in April"
2. **Pure `modified_at`** — "files edited yesterday", "what changed last week"
3. **Pure `event_time`** — "discussions about X around the schema change" (where event_time semantics differ from ctime)
4. **Ambiguous** — "recent work on entity graph" (parser should fall back)
5. **Adversarial** — timezone edges ("yesterday" at 11 PM PST), composition ("early March about RAG"), pre-resolved dates ("on 2026-04-15 about graph migration"), missing-axis (queries on chunks that lack the routed field)

Each row: `{query, axis_expected, window_expected, gold_ids[]}`.

Acceptance bar:
- Wave 1 temporal gold (existing 22 queries) — **no regression** (R@5 ≥ 0.65, P@5 ≥ 0.23).
- New axis-routed gold (40 non-adversarial) — mean R@5 ≥ 0.5.
- Adversarial bucket (10) — measured but not gated; failures documented in `EXPERIMENTS.md`.

Pin `temporal_now=datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)` for reproducibility.

## Atoms (sequenced)

| # | Atom | Output | Test |
|---|---|---|---|
| 1 | SIGNALS schema update + axis docs | `docs/dev/SIGNALS.md` updated; `temporal_axis` enum added to `metadata_filter.py` | Schema doctest passes |
| 2 | Filesystem reader stamps `created_at` + `modified_at` | `_build_index_from_documents` reads ctime/mtime | Unit test: build small folder, assert both fields present |
| 3 | Calendar reader split — keep `event_time`=start, add `created_at`+`modified_at` from event metadata | `apps/calendar_rag.py` or builder updates | Mocked SQLite test |
| 4 | Parser axis routing | `leann/temporal.py`: `parse_temporal_query` returns `(axis, window)` instead of `window` | Frozen-time unit tests, 20 cases across 4 axes |
| 5 | Searcher filter routing + fallback chain + strict mode | `LeannSearcher.search` accepts `temporal_strict`, `temporal_axis` | Integration test: 3-chunk fixture, assert correct axis hits |
| 6 | Specialized reader backfills (email, imessage, browser, wechat, chatgpt, claude, git, slack) | Each reader emits all available axes | Per-reader unit test |
| 7 | Build `context-layer-temporal` + write 50-query agent gold | `tests/eval/context_layer_temporal_gold.jsonl` + `scripts/build_context_layer_temporal.py` | Index meta.json has all axes; gold validates schema |
| 8 | Run baseline (current Wave 1) + treatment (Wave 1.5) on both gold sets | `scripts/eval_temporal.py --multi-axis` outputs table | Acceptance bar met |
| 9 | Close-out: PROGRESS update, SKILL docs updated, DECISIONS-SEARCH.md updated, CHANGELOG entry | Branch ready to merge | Full pytest run (with documented exclusions) |

Estimate: ~2 days serial, ~1 day with /loop parallelism on atoms 3-6.

## /loop driver prompt

```
You are executing Wave 1.5 (multi-axis temporal) from docs/dev/MULTI-AXIS-TEMPORAL-PLAN.md. Branch: feat/multi-axis-temporal (cut from feat/temporal-substrate after merge to main).

Atoms 1-9 are listed in the plan. Execute serially. Each atom:
1. Implement
2. Write unit test
3. Run the relevant test slice + pytest -k temporal
4. Commit with the shape: feat(temporal-axis): <atom title> | fix(temporal-axis): ... | test(temporal-axis): ...
5. Append a docs(dev) entry to PROGRESS.md
6. Move to next atom

Acceptance: atom 8's eval must meet the bar in the plan. If it doesn't, root-cause and iterate on atoms 4-6 before declaring done.

Constraints:
- Embeddings via iq at http://100.122.112.83:8100/v1, model BAAI/bge-m3.
- Don't touch Wave 1 single-axis behavior — fallback chain preserves it.
- temporal_now pinned to 2026-05-19T12:00:00+00:00 in eval scripts.
- Do not modify DiskANN code paths.
- Test exclusions documented in Wave 1 PROGRESS still apply.

Report at each atom close + final eval results.
```

## Decisions (Claude + Codex composite, user-approved 2026-05-19)

1. **Filesystem `event_time`**: **omit**. Stamp only `created_at` (`st_birthtime`) + `modified_at` (`st_mtime`). Don't alias mtime as event_time — schema purity beats query-intuition shortcuts; "recent docs" handled by routing + fallback, not aliasing.
2. **`temporal_strict` default**: **False in production, True in eval.** Plus diagnostics must surface `temporal_strict` value in result metadata so future regressions don't get misread as retrieval quality drops.
3. **kg as Wave 1.5 secondary corpus**: **No — defer entirely to Wave 3.** Wave 1.5 stays focused on axis plumbing.
4. **GCal calendar reader**: **Defer to Wave 3** as the first concrete consumer of the source registry. Existing Apple Calendar reader keeps current behavior; when `created_at` is synthesized from `event_time` (source can't separate them), **mark as synthesized** in chunk metadata (e.g. `created_at_synthesized: true`) so `explain_filters` can surface "this axis is degenerate" instead of pretending it's authoritative.
5. **Fallback priority**: **Symmetric** — routed=event_time → fall back to created_at first; routed=created_at → fall back to event_time first. AND `temporal_axis_fallback_used` exposed in `explain_filters` diagnostics. AND agents can opt into `temporal_strict=True` per-call when the tool invocation came from structured intent.

### Spotlight clarification

Primary mac has Spotlight indexing off — irrelevant. We use `os.stat()` `st_birthtime` + `st_mtime`, both FS-level (HFS+/APFS), Spotlight-independent. Do **not** use `st_ctime` on macOS — it's inode change time, not creation. (Easy bug; documented here so atom 2 doesn't reintroduce it.)

## Out of scope — for later

- Reranking (kg uses Jina cross-encoder — interesting Wave 4 candidate, not multi-axis work).
- Time-of-day filters ("morning", "after work hours"). Wave 1 handles only date granularity.
- Recurring events ("every Monday in March"). Complex; not blocking Wave 3.
- Cross-axis composition in one query ("created last week and edited yesterday"). Possible but rare; needs query AST.
