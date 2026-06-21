---
name: leann-index
description: Build, update, or list LEANN vector indexes (zain's fork — Wave 1 temporal + Wave 2 metadata-aware). Use when the user wants to "index", "build a leann index", "ingest", reindex, force rebuild, use specialized readers (browser, email, calendar, imessage, wechat, chatgpt, claude), or chunk code with AST-aware mode. Embeddings default to BAAI/bge-m3 via iq at http://100.122.112.83:8100/v1.
---

# Building LEANN indexes (zain's fork)

LEANN is a vector DB with graph-based selective recomputation. Index data lives under `.leann/indexes/<name>/`.

**Stack assumptions:**
- Embeddings: `BAAI/bge-m3` via **iq** at `http://100.122.112.83:8100/v1` (Tailscale IP — works from any device on the tailnet). iq routes to local infinity_emb.
- Fallback for hermetic unit tests only: `sentence-transformers/all-MiniLM-L6-v2`.
- Backend: HNSW (default) for large, append-only graph indexes. Use `flat` for small/medium corpora you **edit in place** — exact recall, cheap add/remove/**modify** by stable ID, and no-op rebuilds that embed zero rows. Flat is **not** JSONL-only: it works for `--docs` directory builds (markdown wikis, etc.) just as well. Verified: editing one `.md` file on a flat `--docs` index does a native remove+add of only the changed chunk, no `--force`, no collapse.

## Three entry points

1. **`leann build <name> --docs <paths>`** — generic builder from filesystem.
2. **`leann build-jsonl <name> --input rows.jsonl`** — exact metadata builder for already-chunked rows; use `--incremental-by-id` when rows have stable IDs.
3. **`leann index-<source>`** — specialized reader (browser, email, calendar, imessage, wechat, chatgpt, claude).

## Default invocation (use this unless told otherwise)

```bash
leann build my-docs --docs ./documents/ \
  --embedding-mode openai \
  --embedding-model BAAI/bge-m3 \
  --embedding-api-base http://100.122.112.83:8100/v1 \
  --embedding-api-key iq-local
```

`api-key` value doesn't matter for iq but the OpenAI SDK requires non-empty. Use `iq-local`.

## Code repos

Add AST-aware chunking (function/class-bounded chunks):

```bash
leann build my-code --docs ./src --use-ast-chunking \
  --embedding-mode openai --embedding-model BAAI/bge-m3 \
  --embedding-api-base http://100.122.112.83:8100/v1 \
  --embedding-api-key iq-local
```

Requires `pip install astchunk` in the venv. Falls back to traditional chunking silently if missing — verify by reading `meta.json` after build. Languages supported by astchunk: Python, Java, C#, TS/TSX/JS.

## JSONL / tool-doc indexes

Use `build-jsonl` when the caller already has one row per searchable object and needs exact metadata preserved for filtering/routing. This is the right path for code-mode tool indexes; `leann build --docs ...` preserves reader/file metadata, but markdown files do not carry arbitrary tool fields like `group`, `tier`, or `autoRunnable`.

```jsonl
{"id":"workon","text":"workon switch project repo context","metadata":{"name":"workon","group":"project","tier":"hot","autoRunnable":false}}
```

```bash
leann build-jsonl code-mode-tools \
  --input .pi/code-mode-leann/tools.jsonl \
  --text-field text \
  --metadata-field metadata \
  --id-field id \
  --incremental-by-id \
  --backend-name flat \
  --no-recompute \
  --no-compact \
  --embedding-mode openai \
  --embedding-model BAAI/bge-m3 \
  --embedding-api-base http://100.122.112.83:8100/v1 \
  --embedding-api-key iq-local
```

`--incremental-by-id` requires every row to have a unique stable ID in `--id-field`. If `metadata.id` is also present it must match the top-level ID; LEANN writes the stable ID into passage metadata as `id` and `source_document_id`.

`build-jsonl` persists `source_kind=jsonl`, `input`, `text_field`, `metadata_field`, `id_field`, and `incremental_by_id` into `build_config`, so `leann rebuild code-mode-tools` replays the same JSONL build. It also stores a sidecar of per-row hashes; volatile `metadata.indexed_at` is ignored, so refresh timestamps alone do not re-embed rows.

Incremental behavior is set by the **backend**, and applies to **both** `--docs` directory builds and `build-jsonl` (the stable ID is the JSONL `--id-field` for `build-jsonl`, or the per-file `source_document_id` written to `documents.ids.txt` / `documents.flat_id_map.json` for `--docs`):
- **Flat**: exact NumPy vector matrix with stable-ID add/remove/modify. The preferred backend for small/medium corpora you edit in place (markdown wikis, CRM/tool-doc indexes) where row counts are modest and recall should be exact. Changed rows/files rewrite only the affected array rows via remove-then-add; no-op rebuilds embed zero rows when the content-addressed cache and rowhash sidecar both hit. (Verified on a `.md` `--docs` build: a modified file logs `Incremental native update (~1 modified): removing 1 old chunks, adding 1 new chunks` and the rest of the index is untouched.)
- **IVF + non-compact**: true add/remove/modify by stable ID. Changed rows are remove-then-add, and passage store, offsets, meta, native index, and BM25 sidecar roll back together on failure.
- **HNSW + non-compact**: **add-only** incremental. Added rows preserve stable IDs in `documents.ids.txt` and update BM25; **changed or removed** rows cannot be applied (HNSW can't delete by ID) and require `--force`. Footgun: on a `--no-recompute` HNSW index a plain (non-`--force`) rebuild after editing a file does NOT do a true full rebuild — it loads only the changed files and **silently collapses** the index to just those. Always `--force` HNSW after edits, or build the index as `flat`/`ivf`.
- **Compact/read-only indexes**: no incremental update; use a full rebuild.

Build-time embeddings use LEANN's content-addressed cache by default (`~/.leann/embed-cache.sqlite`, override with `LEANN_EMBED_CACHE_PATH`, disable with `LEANN_EMBED_CACHE=0`). Cache keys include canonicalization version, model, mode/provider, dimensions, normalization/pooling, prompt template, real truncation settings, and model revision when known; operational knobs such as batch size and timeouts intentionally do not affect the key.

If an embedding server swaps weights behind the same model name, set `LEANN_EMBED_MODEL_REV=<stable-revision>` before building so old cache rows miss lazily instead of reusing stale vectors. LEANN also best-effort probes OpenAI-compatible `/models` once per process and uses explicit revision/fingerprint fields when the server exposes them; env wins over probe.

Drift guards force a full rebuild when sidecars no longer agree: duplicate passage IDs, offset/passages mismatch, rowhash ID mismatch, stale BM25 text for the same ID set, missing HNSW native index, or reordered/missing `documents.ids.txt`.

BM25/FTS5 is built and maintained by default. There is no `--prebuild-bm25` flag to pass for current fork behavior; use search-side `--vector-weight 0.0` for pure BM25 or `0.3`-`0.7` for hybrid.

## Force full rebuild

By default `leann build` is **incremental**, and what that covers depends on the backend (see the table above): **HNSW** adds new files only (modify/remove need `--force`); **flat**/**IVF** add, modify, and remove in place by stable ID. To force a full rebuild from scratch on any backend:

```bash
leann build my-docs --docs ./documents --force
```

The content-addressed embedding cache (`~/.leann/embed-cache.sqlite`) makes even a `--force` rebuild cheap — unchanged chunks re-embed for free, so only genuinely changed text hits the embedding server.

## Rebuild with stored config

`leann rebuild <name>` re-runs a build using the config stored with the index (docs paths or JSONL input, embedding flags, chunking/field settings) — no need to re-type the original flags. Incremental delta by default; `--force` for a full rebuild from scratch (upstream #326).

```bash
leann rebuild my-docs            # incremental, reusing stored config
leann rebuild my-docs --force    # full rebuild from scratch
```

Use this instead of remembering the original `leann build ...` invocation. (Differs from `build --force`, which still requires you to re-supply `--docs` and embedding flags.)

## Specialized readers

Each emits canonical SIGNALS metadata (`created_at`, `modified_at`, `event_time`, `indexed_at`, `source_type`, `author`, etc. when available) so Wave 1/1.5 temporal + Wave 2 metadata-aware filters work downstream.

```bash
# Chrome/Brave browser history (sqlite-backed)
# Upstream #315: emits url, domain, last_visited, visit_count, typed_count, title
leann index-browser chrome

# Apple Mail (.emlx)
leann index-email

# Apple Calendar:
# emits event_time (UTC ISO), event_time_local, created_at/modified_at when source columns exist,
# synthesized created_at_synthesized/modified_at_synthesized markers otherwise,
# source_type=calendar, source_id
leann index-calendar --max-count 1000

# iMessage (chat.db)
leann index-imessage

# Imports
leann index-chatgpt --export-path ~/chatgpt-export.zip
leann index-claude  --export-path ~/claude-export.zip
leann index-wechat
```

**All `index-*` commands share the same embedding flags as `leann build`** — pass `--embedding-mode openai --embedding-api-base http://100.122.112.83:8100/v1` etc.

**Wave 1/1.5 guarantee:** every chunk from `index-*` (and `build`) is stamped with `indexed_at` (UTC ISO) at ingest. Filesystem-backed build paths also stamp `created_at` from `st_birthtime` and `modified_at` from `st_mtime` where available; do not treat filesystem `event_time` as meaningful unless a source reader explicitly emits it.

## Watch mode

```bash
leann watch my-docs --docs ./documents/
```

Rebuilds incrementally on FS changes.

## List / remove

```bash
leann list                 # all indexes (local + global)
leann remove my-docs       # local first, then global
```

## All `leann build` flags (reference)

### Positional
- `index_name` — defaults to current directory name.

### Input
- `--docs PATH [PATH …]` — files or directories
- `--file-types ext,ext` — comma-separated extensions filter
- `--include-hidden` / `--no-include-hidden` — dotfiles (default off)

## `leann build-jsonl` flags

- `index_name`
- `--input FILE.jsonl` — required JSONL source
- `--text-field FIELD` — searchable text field (default `text`)
- `--metadata-field FIELD` — metadata object field (default `metadata`)
- `--id-field FIELD` — stable passage ID field (default `id`)
- `--incremental-by-id` — diff JSONL rows by stable ID and re-embed only new/changed rows where the backend can update in place
- plus the same backend and embedding flags as `leann build`

### Embeddings (override defaults above)
- `--embedding-mode {sentence-transformers,openai,mlx,ollama}` — `openai` for iq
- `--embedding-model NAME` — `BAAI/bge-m3` for iq
- `--embedding-host URL` — Ollama-compatible
- `--embedding-api-base URL` — `http://100.122.112.83:8100/v1` for iq
- `--embedding-api-key KEY` — `iq-local` placeholder for iq
- `--embedding-prompt-template "query: "` — prepend to texts (asymmetric models like Qwen3-Embedding)
- `--query-prompt-template "search: "` — separate query prompt

### Chunking
- `--doc-chunk-size N` — default 256 tokens
- `--doc-chunk-overlap N` — default 128 tokens
- `--code-chunk-size N` — default 512 tokens
- `--code-chunk-overlap N` — default 50 tokens
- `--use-ast-chunking` — astchunk function/class boundaries
- `--ast-chunk-size N` — default 300 chars (non-whitespace)
- `--ast-chunk-overlap N` — default 64 chars
- `--ast-fallback-traditional` — fallback if astchunk import fails (default on)

### Lifecycle
- `--force` / `-f` — full rebuild (otherwise incremental)
- `--num-threads N` — parallelism

## Decision guide

Deeper reference: `docs/dev/DECISIONS-INDEX.md`.

### Backend — pick by whether the corpus is edited and how big it is

HNSW is the default, but it is **append-only**: if the corpus gets *edited* (pages rewritten, rows changed), prefer `flat` (small/medium) or `ivf`.

| Use | When |
|---|---|
| **HNSW** (default) | Large, static or **append-only** corpus, fits in RAM. Modify/remove require `--force`. |
| **flat** | Small/medium corpus you **edit in place** (≲100k chunks): exact brute-force recall + true add/remove/**modify** by stable ID, no graph rebuild. Works for `--docs` (markdown) **and** JSONL. Best default for an actively-maintained wiki/tool-doc index. |
| **IVF** | Need true in-place add/remove/modify at larger scale than flat wants to brute-force. FAISS IVF + DirectMap.Hashtable. |
| **DiskANN** | Corpus much larger than RAM (10M+ chunks). Slower build, larger on-disk graph, but searchable from disk. Append-only like HNSW. |

### `--recompute` vs `--no-recompute` (this is THE LEANN tradeoff)

LEANN's value prop: store a pruned graph + recompute embeddings on demand → ~97% storage reduction vs storing all vectors.

| Mode | What's stored | Search latency | Storage | When to use |
|---|---|---|---|---|
| `--recompute` (default for HNSW build) | Graph only; embeddings recomputed via ZMQ embedding server during search | Higher (embedding compute on hot path) | **Tiny** | Big corpora, disk-constrained, you have a fast embedding server (iq). **Default for HNSW.** |
| `--no-recompute` | Graph + all passage embeddings | Low (no recompute) | Large | Small corpus (<100k chunks), or no embedding server available at search time, or latency-critical UI. |

Build and search must agree on recompute, but you rarely set it by hand at search time: since Wave 2.1, `LeannSearcher` auto-detects `recompute_embeddings` from the index `meta.json` (default `None` = read from meta). With `--no-recompute` indexes only the query is embedded at search time; stored passage vectors are reused. Explicit `recompute_embeddings=True/False` still wins if you pass it.

### `--compact` (HNSW only)

Strips per-passage embeddings post-build and freezes the graph. Even smaller on disk, but **read-only** (no `update_index`, no `--force`-less rebuild). Use for archived corpora you'll never touch again. Skip otherwise.

### Chunk size

Default `--doc-chunk-size 256 --doc-chunk-overlap 128` (tokens) is right for prose. Bump for these:

| Corpus | Suggested |
|---|---|
| Long-form essays, books | 512 / 128 |
| Slack/chat (short messages) | Use the dedicated `index-*` reader — chunks are message-grouped, not token-sliced |
| Tables / structured data | 256 / 0 (no overlap, rows are atomic) |
| Code | Prefer `--use-ast-chunking` (function/class boundaries). Falls back to `--code-chunk-size 512 / --code-chunk-overlap 50`. |

**Hard ceiling: bge-m3's context is 8192 tokens.** Chunks longer than that are truncated at embed time (the tail is dropped), so don't set `--doc-chunk-size` above ~7000. Truncation counts tokens with the model's **own** tokenizer (bge-m3 SentencePiece, not OpenAI's cl100k), so the limit is honored exactly. bge-m3 tokenizes ~1.16× denser than cl100k, so a chunk that "looks like" 8000 tokens can be well over the real ceiling.

This truncation is **central and universal** (fixed 2026-06): it lives in `compute_embeddings`, so it applies to **every** embedding mode (`openai`, `ollama`, `mlx`) and every build/recompute path — not just some. There's nothing per-build to configure.

> If you previously dialed chunk size down or set any token-limit override **just to dodge the iq 400/413**, that workaround is obsolete — pick chunk size for retrieval quality, not to avoid the crash. Custom scripts that embed *outside* LEANN (POST straight to a shim) should clip first: `from leann import truncate_for_model` → `rows = truncate_for_model(rows, "BAAI/bge-m3")`.

**Don't lower batch size to avoid OOM.** Embedding-server OOMs (Metal `metal::malloc`) come from `batch_size × padded-max-tokens`, not from any single chunk. The iq shim now enforces a padded-token budget and **splits incoming batches internally**, so `LEANN_EMBED_BATCH_SIZE` is just a throughput/load knob — leave it at the default; you don't need to shrink it to keep the shim alive.

### AST chunking — when

- ✅ Python, Java, C#, TS/TSX/JS (astchunk-supported languages)
- ❌ Other languages — falls back silently to traditional. Pointless flag.
- Verify after build: `cat .leann/indexes/<name>/meta.json | grep ast` — should show `"use_ast_chunking": true`.

### Embedding prompt template (asymmetric models)

| Model | Need prompt template? |
|---|---|
| `BAAI/bge-m3` | No (default iq model — skip the flag) |
| `Qwen3-Embedding-*` | **Yes** — `--embedding-prompt-template "passage: " --query-prompt-template "query: "` |
| `intfloat/e5-*` | Yes — same shape, different prefixes |

If unsure, check the model card. Using wrong prompts silently degrades recall ~10-20%.

## Wave 1 / Wave 2 specifics

**SIGNALS metadata** — for any custom reader, emit chunks with these reserved fields:
- `event_time` (UTC ISO 8601) — when the underlying event happened
- `indexed_at` (UTC ISO 8601) — set automatically at ingest
- `source_type` — enum string (`slack`, `document`, `git_commit`, `calendar`, `email`, etc.)
- `source_id` — stable per-source identifier
- `source_document_id` — stable doc ID (drives `diversify_by` + `context_window` at search time)
- `chunk_seq` — int sequence within `source_document_id`
- `author`, `participant_ids` — identity
- `parent_ref` — containing structure (thread, channel, PR)
- `mentioned_urls`, `mentioned_refs`, `mentioned_files` — referential signals

Full schema in `docs/dev/SIGNALS.md`.

## Common gotchas

1. **HNSW can't modify or remove incrementally.** With `--recompute`, an incremental rebuild raises `ValueError` early (fork QoL). With `--no-recompute`, editing an existing file is worse: a plain rebuild **silently collapses** the index to just the changed files (the "falling back to full rebuild" message is misleading — it only loads the changed docs). Always `--force` HNSW after edits, or build the index as `--backend-name flat`/`ivf` so edits apply in place. (Pure *adds* of new files are fine on HNSW without `--force`.)
2. **`--compact` indexes are read-only** — no `update_index`, no incremental add.
3. **`--use-ast-chunking` silently falls back** if `astchunk` isn't installed. Always verify by checking the index `meta.json`.
4. **Chunk size is in TOKENS** for doc/code, **CHARACTERS** for AST. Don't conflate.
5. **iq must be reachable.** Check `curl -sf http://100.122.112.83:8100/v1/models` before kicking off a long build.

## After building — programmatic access

```python
from leann import LeannSearcher
# Bare name resolves to .leann/indexes/<name>/documents.leann (fork QoL)
s = LeannSearcher("my-docs")
```

See the `leann-search` skill for the search surface.

## When the user asks

- *"build an index"* / *"index this folder"* → `leann build <name> --docs <path>` with the iq embedding flags
- *"index my email/calendar/messages/browser"* → `leann index-<source>` (same embedding flags)
- *"reindex"* / *"rebuild"* → `leann rebuild <name>` (reuses stored config); `--force` for full rebuild
- *"rebuild but I forgot the original flags"* → `leann rebuild <name>` (config is stored with the index)
- *"index code"* → add `--use-ast-chunking`
- *"index won't update"* / *"pages went missing / search returns almost nothing after a rebuild"* → on **HNSW** a *modified or removed* file isn't applied incrementally, and a non-`--force` rebuild can silently collapse the index to just the changed files. Fix: `--force`, or rebuild the index as `--backend-name flat` (small/medium) or `ivf` so edits modify in place. Also rule out `--compact` (read-only).
- *"use a specific model"* → override `--embedding-model` + `--embedding-mode`
