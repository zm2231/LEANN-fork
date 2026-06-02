---
name: leann-index
description: Build, update, or list LEANN vector indexes (zain's fork — Wave 1 temporal + Wave 2 metadata-aware). Use when the user wants to "index", "build a leann index", "ingest", reindex, force rebuild, use specialized readers (browser, email, calendar, imessage, wechat, chatgpt, claude), or chunk code with AST-aware mode. Embeddings default to BAAI/bge-m3 via iq at http://100.122.112.83:8100/v1.
---

# Building LEANN indexes (zain's fork)

LEANN is a vector DB with graph-based selective recomputation. Index data lives under `.leann/indexes/<name>/`.

**Stack assumptions:**
- Embeddings: `BAAI/bge-m3` via **iq** at `http://100.122.112.83:8100/v1` (Tailscale IP — works from any device on the tailnet). iq routes to local infinity_emb.
- Fallback for hermetic unit tests only: `sentence-transformers/all-MiniLM-L6-v2`.
- Backend: HNSW (default). Don't touch DiskANN/IVF unless explicitly asked — they're pre-existing surfaces, not part of Wave 1/2.

## Two entry points

1. **`leann build <name> --docs <paths>`** — generic builder from filesystem.
2. **`leann index-<source>`** — specialized reader (browser, email, calendar, imessage, wechat, chatgpt, claude).

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

## Force full rebuild

By default `leann build` is **incremental** — adds new files only. To rebuild from scratch:

```bash
leann build my-docs --docs ./documents --force
```

## Rebuild with stored config

`leann rebuild <name>` re-runs a build using the config stored with the index (docs paths, embedding flags, chunking) — no need to re-type the original flags. Incremental delta by default; `--force` for a full rebuild from scratch (upstream #326).

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

### Backend (default is HNSW — don't change unless one of these applies)

| Use | When |
|---|---|
| **HNSW** (default) | Static or append-only corpus, fits in RAM. 95% of cases. |
| **IVF** | Need true in-place add **and remove** without rebuild (e.g. live-syncing Slack/email where messages get edited or deleted). FAISS IVF + DirectMap.Hashtable. |
| **DiskANN** | Corpus much larger than RAM (10M+ chunks). Slower build, larger on-disk graph, but searchable from disk. |

### `--recompute` vs `--no-recompute` (this is THE LEANN tradeoff)

LEANN's value prop: store a pruned graph + recompute embeddings on demand → ~97% storage reduction vs storing all vectors.

| Mode | What's stored | Search latency | Storage | When to use |
|---|---|---|---|---|
| `--recompute` (default for HNSW build) | Graph only; embeddings recomputed via ZMQ embedding server during search | Higher (embedding compute on hot path) | **Tiny** | Big corpora, disk-constrained, you have a fast embedding server (iq). **Default for HNSW.** |
| `--no-recompute` | Graph + all passage embeddings | Low (no recompute) | Large | Small corpus (<100k chunks), or no embedding server available at search time, or latency-critical UI. |

Build flag and search flag **must match**. Mismatched search will error or silently return garbage.

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

1. **HNSW + recompute + incremental rebuild** raises `ValueError` early (fork QoL). Use `--force` for full rebuild.
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
- *"index won't update"* → it's compact or HNSW+recompute; use `--force`
- *"use a specific model"* → override `--embedding-model` + `--embedding-mode`
