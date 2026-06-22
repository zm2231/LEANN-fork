---
name: leann-index
description: Build, update, rebuild, watch, or list LEANN indexes. Use for `leann build`, `build-jsonl`, `rebuild`, `watch`, source readers, backend choice, incremental behavior, metadata stamping, folder filters, embedding cache, and AST-aware code indexing.
---

# LEANN Indexing

Use this when creating or maintaining indexes under `.leann/indexes/<name>/`.

## First Decision: Backend

Choose by corpus size and whether content is edited in place.

| Backend | Best use | Incremental `--docs` | Incremental `build-jsonl --incremental-by-id` |
|---|---|---|---|
| `flat` | Small/medium edited corpora, exact recall | Add/remove/modify in place | Add/remove/modify in place |
| `ivf` | Larger edited corpora, approximate FAISS search | Add/remove/modify in place | Add/remove/modify in place |
| `hnsw` | Large mostly static corpus that fits in RAM | Any docs change triggers full rebuild | Add-only for non-compact indexes; changed/removed rows rebuild |
| `diskann` | Very large disk-backed corpus | Full rebuild | Full rebuild |
| compact indexes | Archived/read-only indexes | No incremental | No incremental |

Default to `flat --no-recompute` for actively edited markdown/wiki/CRM/tool-doc corpora. Default to `hnsw` for large mostly static corpora. Use `ivf` when flat is too slow but remove/modify still matters.

## Embeddings

Pick the embedding provider explicitly for reproducible builds. Examples:

```bash
--embedding-mode sentence-transformers \
--embedding-model facebook/contriever
```

For OpenAI-compatible embedding servers:

```bash
--embedding-mode openai \
--embedding-model <model-name> \
--embedding-api-base <base-url> \
--embedding-api-key <api-key>
```

If the server requires a non-empty placeholder key, pass one explicitly.

## Build From Files

```bash
leann build docs --docs ./documents \
  --backend-name flat \
  --no-recompute \
  --embedding-mode sentence-transformers \
  --embedding-model facebook/contriever
```

`build --docs` preserves file/path metadata and stamps:

- `indexed_at`
- `created_at`
- `modified_at`
- `source_root`
- `relative_path`
- `folder_path`
- `top_folder`

### Folder Filters

Use `--top-folder-depth` when the first path segment is only a wrapper folder.

```bash
leann build docs --docs /path/to/docs \
  --top-folder-depth 2 \
  --backend-name flat \
  --no-recompute \
  --embedding-mode sentence-transformers \
  --embedding-model facebook/contriever
```

For `Clients/BD/proposal.md` with `--top-folder-depth 2`, metadata is:

- `relative_path = Clients/BD/proposal.md`
- `folder_path = Clients/BD`
- `source_root = <docs root>`
- `top_folder = BD`

Then search with:

```bash
leann search dropbox "proposal" \
  --metadata-filters '{"top_folder":{"==":"BD"}}'
```

Do not call this `category` in core LEANN. `top_folder` is factual filesystem metadata; app-specific category semantics belong in JSONL metadata or a source reader.

## Build From JSONL

Use `build-jsonl` when the caller already has one row per searchable object and needs exact metadata preserved.

```jsonl
{"id":"workon","text":"workon switch project repo context","metadata":{"name":"workon","group":"project","tier":"hot"}}
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
  --embedding-mode sentence-transformers \
  --embedding-model facebook/contriever
```

Rules:

- Every row needs a unique stable `--id-field`.
- If `metadata.id` exists, it must match the top-level ID.
- LEANN writes the stable ID into metadata as `id` and `source_document_id`.
- `metadata.indexed_at` is ignored for rowhashes, so timestamp refreshes do not re-embed unchanged rows.

## Rebuild And Watch

```bash
leann rebuild docs
leann rebuild docs --force
leann watch docs --interval 30
```

`rebuild` replays stored `build_config`, including docs paths, JSONL input, chunking fields, backend, embedding flags, `--top-folder-depth`, and `--incremental-by-id`.

Use `--force` for a full rebuild. The content-addressed embedding cache keeps force rebuilds cheap when text is unchanged.

## Code Indexes

```bash
leann build code --docs ./src --use-ast-chunking \
  --backend-name flat \
  --no-recompute \
  --embedding-mode sentence-transformers \
  --embedding-model facebook/contriever
```

AST chunking supports Python, Java, C#, TS/TSX/JS. Other languages fall back to token chunking. Verify by reading `meta.json` after build.

## Source Readers

Specialized readers emit richer source metadata, including `source_type`, `event_time`, `author`, and source-specific IDs when available.

```bash
leann index-browser chrome
leann index-email
leann index-calendar --max-count 1000
leann index-imessage
leann index-chatgpt --export-path ~/chatgpt-export.zip
leann index-claude --export-path ~/claude-export.zip
leann index-wechat --export-dir ~/wechat-export
```

All `index-*` commands share the same embedding flags as `leann build`.

## Important Flags

Input:

- `--docs PATH [PATH ...]`
- `--file-types .md,.pdf,.txt`
- `--include-hidden` / `--no-include-hidden`
- `--top-folder-depth N`

Backend:

- `--backend-name {hnsw,diskann,ivf,flat}`
- `--compact` / `--no-compact`
- `--recompute` / `--no-recompute`
- `--force`

Chunking:

- `--doc-chunk-size N`, `--doc-chunk-overlap N`
- `--code-chunk-size N`, `--code-chunk-overlap N`
- `--use-ast-chunking`
- `--ast-chunk-size N`, `--ast-chunk-overlap N`

JSONL:

- `--input FILE.jsonl`
- `--text-field FIELD`
- `--metadata-field FIELD`
- `--id-field FIELD`
- `--incremental-by-id`

Default `--docs` extensions include documents (`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`), common code/config/markup files, and `.ipynb`. PDFs get PyMuPDF/pdfplumber first, then generic fallback.

## Embedding Cache

Build-time embeddings use `~/.leann/embed-cache.sqlite` by default.

- Override: `LEANN_EMBED_CACHE_PATH=/path/cache.sqlite`
- Disable: `LEANN_EMBED_CACHE=0`
- Force model revision key: `LEANN_EMBED_MODEL_REV=<stable-revision>`

Cache keys include canonicalization version, model, provider/mode, dimensions, normalization/pooling, prompt template, truncation settings, and model revision when known. Batch size and timeouts do not affect cache keys.

BM25/FTS5 is built and maintained by default. Do not pass a `--prebuild-bm25` flag in this fork.

## Shipped: `--top-folder-depth`

SHIP for the requested folder-filter change:

- Added `leann build --top-folder-depth` in `packages/leann-core/src/leann/cli.py`.
- Default is `1`, preserving current behavior.
- With `--top-folder-depth 2`, `Clients/BD/proposal.md` yields `top_folder = BD`.
- Wired through build defaults, explicit option tracking, persisted `build_config`, rebuild, and watch replay.
- Documented in `docs/configuration-guide.md` and `docs/CHANGELOG.md`.
- Verified with `uv run pytest tests/test_cli_build_config.py tests/test_indexed_at.py -q`: `22 passed`.
- Verified with `uv run ruff check packages/leann-core/src/leann/cli.py tests/test_cli_build_config.py tests/test_indexed_at.py`.
- Real CLI smoke confirmed `--metadata-filters '{"top_folder":{"==":"BD"}}'` returned only the BD document.

## Common Pitfalls

- HNSW is not the backend for edited docs. Use `flat` or `ivf`.
- Compact indexes are read-only.
- `build --docs` cannot preserve arbitrary app metadata; use `build-jsonl` for that.
- Filesystem `event_time` and `source_type` are not present unless a source reader or JSONL producer emits them.
- For Qwen/E5-style asymmetric models, pass both document and query prompt templates.
