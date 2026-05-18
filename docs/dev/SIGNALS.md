# SIGNALS — Canonical Chunk Metadata Schema

**Status:** draft, locked for Wave 1 (temporal substrate). Extensions go to a new section, not in-place edits.

**Purpose:** every LEANN reader emits the same field names with the same semantics. Search, filtering, and connectors all key off this schema. Without a fixed schema, "cross-source connections" cannot be implemented downstream — each connector would need a per-reader translation table.

**Reserved keys:** the keys below are reserved. Readers MUST NOT use them for different meanings. Anything outside this list is free-form per-reader (under `metadata.extra`).

---

## Glossary (read first)

- **chunk** — a unit of text indexed by LEANN; carries a `text` body and a `metadata` dict (`SearchResult.metadata`, `packages/leann-core/src/leann/api.py:120-126`).
- **reader** — a module under `apps/` or `packages/leann-core/src/leann/cli.py` that produces chunks from a source (Slack, email, code, calendar, browser).
- **connector** — Wave 6+ component that reads chunks and materializes edges between them based on signals. Out of scope for Wave 1; the schema is designed for it.
- **event_time** — when the underlying real-world event happened (message sent, commit authored, file created). Distinct from `indexed_at`.
- **indexed_at** — when LEANN ingested the chunk. Useful for "what's new in the index since X."
- **canonical** — UTC, ISO 8601 with `T` separator, second precision minimum (`2026-05-15T14:30:00+00:00` or `2026-05-15T14:30:00Z`).

---

## Reserved fields

### Temporal

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `event_time` | ISO 8601 string, UTC | yes for time-bearing sources | `"2026-05-15T14:30:00+00:00"` | When the event happened in the real world. ALWAYS UTC. Naive datetimes are forbidden — reader MUST localize before serializing. |
| `event_time_local` | ISO 8601 string with offset | optional | `"2026-05-15T10:30:00-04:00"` | If the local time matters (e.g., "morning vs evening"), include this in addition to `event_time`. Never as a substitute. |
| `indexed_at` | ISO 8601 string, UTC | yes (stamped by `_build_index_from_documents`) | `"2026-05-15T18:00:00Z"` | Set automatically at ingest. Readers SHOULD NOT set this — the builder does. |

**Rule:** if a source has no meaningful event timestamp (e.g., a reference doc with no creation date), omit `event_time` entirely. Do not fabricate one from `indexed_at` — that breaks temporal queries silently.

### Identity / authorship

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `author` | string | yes if known | `"zain"`, `"alice@example.com"`, `"slack:U01ABC"` | The single entity responsible for creating this chunk's content. Prefer stable identifiers (email, slack ID) over display names. |
| `activity_type` | enum string | yes if `author` set | `"authored"`, `"received"`, `"visited"`, `"modified"`, `"created"` | The relationship between the named author and the chunk. `authored` = wrote it. `received` = it was sent to them. `visited` = they browsed it. |
| `participant_ids` | list[string] | optional | `["zain", "alice@example.com"]` | All people involved (sender + recipients for email, channel members for Slack burst). Includes `author`. |

### Source identification

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `source_type` | enum string | yes | `"document"`, `"git_commit"`, `"slack"`, `"daily_summary"`, `"email"`, `"imessage"`, `"browser_history"`, `"calendar"`, `"code"`, `"voice_memo"`, `"journal"`, `"notion"` | Open enum; new sources add new values. Used for source-typed filtering and per-type extraction rules. `daily_summary` = LLM-generated per-channel-per-day rollups (`event_time` = the day at 00:00 UTC, multi-author so `author=null`). |
| `source_id` | string | yes if stable identifier exists | `"<commit-sha>"`, `"<slack-ts>"`, `"<msg-id>"` | Stable per-source identifier so the same event can be deduplicated across re-indexes. |
| `source_url` | string | optional | `"https://github.com/.../commit/abc"`, `"slack://channel/C01/p123"` | Deep link back to the original. Connectors use this to merge across readers. |
| `project_id` | string | optional | `"leann"`, `"context-layer"` | Project/repo/scope this chunk belongs to. Critical for multi-project corpora. |
| `parent_ref` | string | optional | `"thread:<id>"`, `"pr:#287"`, `"channel:#general"` | Pointer to a containing structure. Lets connectors materialize hierarchical edges (`commit ∈ PR ∈ project`) without LEANN itself knowing the hierarchy. |

### Referential signals (for connectors)

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `mentioned_urls` | list[string] | optional | `["https://github.com/.../issues/45"]` | URLs found in the chunk text. Extracted by reader via regex at ingest. Empty list if none. |
| `mentioned_refs` | list[string] | optional | `["#287", "JIRA-123", "@alice"]` | Issue/ticket/mention references. Free-form strings; connectors decide what to match. |
| `mentioned_files` | list[string] | optional | `["src/api.py", "docs/README.md"]` | File paths found in the chunk text. |

### Free-form extension

| Field | Type | Notes |
|---|---|---|
| `extra` | dict | Per-reader data that doesn't fit above. NOT read by core filtering or connectors. Use sparingly. |

---

## Document hierarchy (Wave 2 additions)

| Field | Type | Required | Example | Notes |
|---|---|---|---|---|
| `source_document_id` | string | recommended | `"sha256-abc123"`, `"hearing_2025_05_20_sfrc_budget"` | Stable identifier for the source document this chunk came from. Multiple chunks from the same PDF, video, or hearing transcript share one `source_document_id`. Drives `diversify_by` and `context_window`. |
| `chunk_seq` | integer | recommended when `source_document_id` is set | `0`, `1`, `2` | Ordering within `source_document_id`. Use sequential integers from 0; gaps are allowed. |
| `event_id` | string | optional | `"hearing_2025_05_20_sfrc_budget"` | Cross-source event identifier. A hearing video, transcript, and press release can share this. Reserved for connector-layer use. |
| `source_url` | string | optional | `"https://example.com/hearing"` | Deep link back to the original source. Repeated here as part of the citation/provenance group. |
| `event_date` | ISO 8601 date or datetime string | optional | `"2025-05-20"` | Human-facing event date when a full `event_time` is not available or when date-level grouping is useful. |
| `speaker` | string | optional | `"Sen. Rubio"` | Speaker or named person responsible for the quoted span, when available. |
| `public_citation_allowed` | boolean | optional | `true` | Whether the app layer may show this chunk as a public citation. |
| `review_required` | boolean | optional | `false` | Whether the app layer should require human review before publishing or citing this chunk. |

---

## What this schema is NOT

- **Not a knowledge graph.** Edges between chunks live in a sidecar (Wave 6+), not in chunk metadata. Don't add `linked_to: [chunk_id, ...]` fields.
- **Not a memory model.** `importance`, `decay_factor`, `ttl`, `consolidated_into` belong in `leann-memory` (separate repo, Wave 2 of the larger plan), not here.
- **Not a privacy classifier.** `privacy_scope` is a context-layer concept; if you need it, put it under `extra.privacy_scope`. Don't reserve a top-level key here.
- **Not a CF-dimension store.** D/P/C/M/E tagging is context-layer-specific; goes under `extra.cf_dimensions`.

---

## Validation

A `tests/test_signals_schema.py` test (Wave 1, atom 8) loads every reader's output for a 10-chunk sample and asserts:
1. `event_time` parses as ISO 8601 with UTC offset if present
2. `source_type` is in the documented enum
3. `activity_type` is in the documented enum if `author` is set
4. `indexed_at` is set by the builder and is UTC ISO
5. No naive datetimes anywhere in metadata
6. `participant_ids` is a list and includes `author` if both are set
