# Index-time decisions — deep reference

Companion to `.claude/skills/leann-index/SKILL.md`. The skill has the quick tables; this file has the *why* and the failure modes.

## Backend choice

### HNSW (default)

FAISS Hierarchical Navigable Small World graph. In-memory, very fast (~ms for top-k=10 on 100k chunks). LEANN's `leann-backend-hnsw` adds graph-only mode where embeddings are recomputed on demand.

**Strengths:** lowest search latency, well-understood, mature. Default for a reason.

**Weaknesses:**
- Incremental add only — modify or remove triggers full rebuild (`--force`).
- Whole graph must fit in RAM.
- HNSW + `--recompute` + incremental was a footgun (silently corrupted `passages.jsonl`). Fork now raises `ValueError` early.

**Pick when:** corpus is static or append-only AND fits in RAM. This is 95% of cases.

### IVF

FAISS `IndexIVFFlat` with `DirectMap.Hashtable`. Quantized clusters; supports `add_vectors` and `remove_ids` in place.

**Strengths:** true delete + modify without rebuild. Critical for live-syncing data (Slack edits, email deletes, file watches).

**Weaknesses:** lower recall than HNSW at same parameter budget. Larger on disk than HNSW + recompute.

**Pick when:** your corpus mutates and you can't afford full rebuilds. iMessage live-sync, Slack with edits/redactions, browser history with retention windows.

### DiskANN

Graph index designed to live on SSD, not in RAM.

**Strengths:** scales beyond memory. Good for 10M+ chunk corpora.

**Weaknesses:**
- Build is slow.
- Native code SIGSEGVs on tiny test fixtures (<256 vectors) — pre-existing macOS issue. Tests excluded in Wave 1 close-out.
- `_run_build` had a spawn-mode pickle bug — fixed in `837a4c2`.

**Pick when:** corpus is genuinely larger than RAM AND HNSW + recompute isn't enough.

## Recompute — the LEANN tradeoff

The whole point of LEANN: store a pruned graph, recompute embeddings on demand. ~97% storage reduction.

### How it works

At search time, the searcher needs the embedding for each graph node it visits. With `--recompute`:
1. Read the passage text from `passages.jsonl` (via offset map).
2. Send it to the ZMQ embedding server.
3. Score against the query embedding.

Without `--recompute`, step 2 is a lookup in pre-stored embeddings.

### Cost model

| | `--recompute` | `--no-recompute` |
|---|---|---|
| Storage per 1M 1024-dim chunks | ~50 MB (graph only) | ~4 GB (embeddings + graph) |
| Search latency (top-k=10, complexity=64) | 50-200ms (depends on embedding server) | 5-20ms |
| Embedding server required at search time | Yes | No |
| Build time | Same | Same |

### Decision

- Disk is cheap. **Latency is what kills UX.** If your embeddings fit, `--no-recompute` is usually right for interactive use.
- LEANN's design assumes you have a fast local embedding server (iq with infinity_emb on MPS/CPU). With iq at 100.122.112.83:8100, recompute overhead is ~30-80ms — acceptable.
- For batch / overnight retrieval, `--recompute` is free.

### `--compact`

Goes a step further than `--recompute`: strips per-node embeddings entirely and freezes the graph topology. Read-only after that point. Only sensible for archived corpora you'll truly never update.

## Chunking

### Token-based (`--doc-chunk-size`, `--code-chunk-size`)

Splits on token count (using the embedding model's tokenizer or tiktoken as fallback). Overlap pads each chunk with the tail of the previous one.

- **Default 256/128** for documents — balances recall (small enough to be a focused unit) with context (overlap catches cross-chunk references).
- **Bump to 512/128** for long-form prose with cross-paragraph dependencies (essays, book chapters).
- **Drop overlap to 0** for atomic units (table rows, CSV records, calendar events one-per-chunk).

### AST chunking (`--use-ast-chunking`)

Uses `astchunk` (requires `pip install astchunk`) to split on function/class boundaries. Sizes in **characters**, not tokens — `--ast-chunk-size 300` is non-whitespace chars.

- ✅ Python, Java, C#, TypeScript/TSX/JavaScript
- ❌ Anything else falls back silently. Always verify post-build:
  ```bash
  jq '.use_ast_chunking, .ast_chunk_size' .leann/indexes/<name>/meta.json
  ```

### Specialized readers

For Slack/email/calendar/iMessage/browser/wechat/chatgpt/claude — use `leann index-<source>` rather than `leann build`. The reader handles message boundaries, threading, attachment expansion. Token-based chunking on a chat log produces garbage.

## Embedding prompt templates

Asymmetric embedding models distinguish "is this a passage?" from "is this a query?" with prefix prompts. Using the wrong prompt (or none) on an asymmetric model silently costs 10-20% recall.

| Model | Build (passage) prefix | Query prefix |
|---|---|---|
| `BAAI/bge-m3` (iq default) | (none) | (none) |
| `Qwen3-Embedding-0.6B/8B` | `"passage: "` | `"query: "` |
| `intfloat/e5-large-v2` | `"passage: "` | `"query: "` |
| `sentence-transformers/all-MiniLM-L6-v2` (test fallback) | (none) | (none) |

The build flag `--embedding-prompt-template` is **stored in meta.json**. The searcher reads it back automatically — you don't need to repeat it at search time (passages reuse the build prefix). But `--query-prompt-template` is separate and goes on search-side calls.

## SIGNALS metadata

For custom readers, emit chunks with these reserved fields so Wave 1 temporal + Wave 2 metadata filtering work:

| Field | Type | Required | Notes |
|---|---|---|---|
| `event_time` | UTC ISO 8601 | If temporal sense exists | Drives `enable_temporal=True` parsing |
| `indexed_at` | UTC ISO 8601 | Auto-stamped | Don't set manually |
| `source_type` | enum string | Yes | `slack`, `document`, `git_commit`, `calendar`, `email`, `whatsapp`, ... |
| `source_id` | string | Yes | Stable per-source identifier |
| `source_document_id` | string | Recommended | Drives `diversify_by` + `context_window` |
| `chunk_seq` | int | Recommended | Sequence within `source_document_id` |
| `author` | string | If applicable | |
| `participant_ids` | list[string] | If applicable | |
| `parent_ref` | string | If applicable | Thread, channel, PR id |
| `mentioned_urls` | list[string] | If applicable | Substring-filterable |
| `mentioned_refs` | list[string] | If applicable | Code/issue refs |
| `mentioned_files` | list[string] | If applicable | |

Full schema: `docs/dev/SIGNALS.md`.

## Incremental build — what's safe

`leann build` is incremental by default. Re-running on an existing index applies the minimal delta.

| Backend | Add | Modify | Remove |
|---|---|---|---|
| HNSW (non-compact, no recompute) | ✅ | ❌ → use `--force` | ❌ → use `--force` |
| HNSW + recompute | ❌ — raises ValueError (fork QoL) | ❌ | ❌ |
| HNSW + compact | ❌ — read-only | ❌ | ❌ |
| IVF | ✅ | ✅ (remove+add) | ✅ |
| DiskANN | ✅ | ❌ → use `--force` | ❌ → use `--force` |

`--force` rebuilds from scratch regardless. Use it whenever modify/remove is needed on a non-IVF backend.

## Common gotchas (consolidated)

1. **HNSW + recompute + incremental** → `ValueError`. Use `--force`.
2. **`--compact` is read-only** — no updates, no `--force`-less rebuild.
3. **`--use-ast-chunking` silently falls back** when astchunk isn't installed or language unsupported. Verify via meta.json.
4. **Chunk size units**: tokens for `--doc-chunk-size`/`--code-chunk-size`, **characters** for `--ast-chunk-size`. Don't conflate.
5. **iq reachable**: `curl -sf http://100.122.112.83:8100/v1/models` before long builds.
6. **Wrong embedding prompt on asymmetric model**: silent 10-20% recall loss. Read the model card.
7. **Token-chunking chat logs**: produces garbage. Use the specialized `index-<source>` reader.
