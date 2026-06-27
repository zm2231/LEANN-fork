# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

LEANN is a lightweight vector database and RAG (Retrieval-Augmented Generation) system that achieves 97% storage reduction compared to traditional vector databases through graph-based selective recomputation. It enables semantic search across various data sources (emails, browser history, chat history, code, documents) on a single laptop without cloud dependencies.

## Build & Development Commands

### Quick install (pip)

```bash
pip install leann
```

### Development setup (from source)

```bash
# Install uv first (required package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

git submodule update --init --recursive

# macOS
brew install libomp boost protobuf zeromq pkgconf
uv sync

# Ubuntu/Debian
sudo apt-get install libomp-dev libboost-all-dev protobuf-compiler \
    libabsl-dev libmkl-full-dev libaio-dev libzmq3-dev
uv sync

# Windows (requires VS 2022 Build Tools with C++ workload, vcpkg, chocolatey)
choco install cmake swig pkgconfiglite nuget.commandline -y
vcpkg install zeromq:x64-windows openblas:x64-windows lapack:x64-windows boost-program-options:x64-windows protobuf:x64-windows
# Set CMAKE_PREFIX_PATH, PKG_CONFIG_PATH, OPENBLAS_LIB to vcpkg paths (see README)
uv sync --extra diskann

# Install lint tools
uv sync --group lint

# Install test tools
uv sync --group test
```

## Code Quality

```bash
# Format code
ruff format

# Lint with auto-fix
ruff check --fix

# Pre-commit hooks (install once)
pre-commit install

# Run pre-commit manually
uv run pre-commit run --all-files
```

## Architecture

### Core API Layer (`packages/leann-core/src/leann/`)

- `api.py`: Main APIs - `LeannBuilder`, `LeannSearcher`, `LeannChat`
- `react_agent.py`: `ReActAgent` for multi-turn reasoning
- `cli.py`: CLI implementation (`leann build`, `leann search`, `leann ask`)
- `chat.py`: LLM provider integrations (OpenAI, Ollama, HuggingFace, Anthropic)
- `embedding_compute.py`: Embedding computation (sentence-transformers, MLX, OpenAI)
- `metadata_filter.py`: Search result filtering by metadata

### Backend Layer (`packages/`)

- `leann-backend-hnsw/`: Default backend using FAISS HNSW for fast in-memory search
- `leann-backend-ivf/`: IVF backend (FAISS IndexIVFFlat + DirectMap.Hashtable) supporting in-place add/remove without rebuild
- `leann-backend-flat/`: Exact NumPy flat-vector backend with stable-ID add/remove for small/medium indexes
- `leann-backend-diskann/`: DiskANN backend for larger-than-memory datasets
- `leann-mcp/`: MCP server for Codex integration

Backends are auto-discovered via `leann-backend-*` naming convention and registered in `registry.py`.

### RAG Applications (`apps/`)

Example applications demonstrating RAG on various data sources:
- `document_rag.py`: PDF/TXT/MD documents
- `email_rag.py`: Apple Mail
- `browser_rag.py`: Chrome browser history
- `wechat_rag.py`, `imessage_rag.py`: Chat history
- `code_rag.py`: Codebase search with AST-aware chunking
- `slack_rag.py`, `twitter_rag.py`: MCP-based live data

## Key Design Patterns

### Incremental Update (IVF backend)

The IVF backend supports in-place updates and deletes without rebuilding the entire index:
- `add_vectors(index_path, embeddings, passage_ids)`: Append new vectors to an existing index.
- `remove_ids(index_path, passage_ids)`: Remove vectors by passage ID using FAISS DirectMap.Hashtable.
- `LeannBuilder.update_index()`: High-level API that orchestrates remove-then-add for changed files, compacts `passages.jsonl`, and updates the offset map.

`leann build` is idempotent — re-running it on an existing index automatically performs an incremental update instead of a full rebuild. It detects new, modified, and removed files and applies the minimal set of changes:
- **IVF / flat**: Support add, remove, and modify incrementally (remove old chunks then re-insert).
- **HNSW**: Does not support docs incremental updates; any added, modified, or removed docs trigger a full rebuild.
- Use `--force` / `-f` to force a full rebuild regardless.

### Index Structure

A LEANN index consists of:
- `<name>.meta.json`: Metadata (backend, embedding model, dimensions)
- `<name>.passages.jsonl`: Raw text chunks with metadata
- `<name>.passages.idx`: Offset map for fast passage lookup
- `<name>.index`: Backend-specific vector index

### Embedding Recomputation

The core storage optimization: instead of storing embeddings, LEANN stores a pruned graph and recomputes embeddings on-demand during search via ZMQ server communication.

## CLI Usage

```bash
# Build index
leann build my-docs --docs ./documents/

# Search
leann search my-docs "query"

# Interactive chat
leann ask my-docs --interactive

# List indexes
leann list

# Remove index
leann remove my-docs
```

## Custom Fork — `zain-canonical` branch

This repo is maintained as a reference fork. Local patches live on `zain-canonical`
on top of upstream `main`.

### Install from this branch

Always install `leann-core` and the local backends from this checkout together.
Do not install `leann-core` from the fork while leaving `leann-backend-hnsw`
from PyPI: filtered no-recompute search depends on the fork's HNSW stored-vector
subset scoring (`HNSWSearcher.score_passage_ids`). A stale backend silently falls
back to expensive passage re-embedding.

Canonical local CLI reinstall command:

```bash
repo_root="$(pwd)"
uv tool install --reinstall \
  "${repo_root}/packages/leann-core" \
  --with "${repo_root}/packages/leann-backend-hnsw" \
  --with "${repo_root}/packages/leann-backend-ivf" \
  --with "${repo_root}/packages/leann-backend-flat" \
  --with "${repo_root}/packages/leann-backend-diskann"
```

Also refresh regular venvs that import LEANN:

```bash
repo_root="$(pwd)"
uv pip install --python "${repo_root}/.venv/bin/python" \
  -e "${repo_root}/packages/leann-core" \
  -e "${repo_root}/packages/leann-backend-hnsw" \
  -e "${repo_root}/packages/leann-backend-ivf" \
  -e "${repo_root}/packages/leann-backend-flat" \
  -e "${repo_root}/packages/leann-backend-diskann"
```

Local helper scripts for personal machines must stay gitignored. This checkout
uses `docs/dev/install-local-leann.sh` locally for the full reinstall procedure
and source-hash/import checks; recreate it from the commands above if it is
absent.

```bash
docs/dev/install-local-leann.sh
```

Pass additional Python interpreters to update more venvs:

```bash
docs/dev/install-local-leann.sh /path/to/venv/bin/python
```

### Keeping up with upstream

```bash
git fetch origin
git rebase origin/main   # on zain-canonical
# Reinstall after rebase
docs/dev/install-local-leann.sh
```

### Patches applied (on top of upstream)

| Commit | Description |
|--------|-------------|
| `fix(api)` | `LeannSearcher` falls back to `.leann.meta.json` extension when `.meta.json` not found |
| `perf(list)` | Skip home-dir rglob when cwd isn't a leann project; cache discovery results (was scanning each project twice); replace `rglob` with `os.walk` that skips `node_modules`, `.git`, `.leann`, `.worktrees`, `.venv`, and other heavy dirs; suppress spurious "build your first index" tip when other projects exist |

## Common Development Tasks

Running example RAG applications:
```bash
# Document RAG (easiest to test)
python -m apps.document_rag --query "What is LEANN?"

# Code RAG
python -m apps.code_rag --repo-dir ./src --query "How does search work?"
```

## Python Version

Requires Python 3.10+ (uses PEP 604 union syntax `X | Y`).




# Agent Coding Guidelines

## General
- Voice input may contain typos — interpret intent, not literal text.
- When you encounter a problem, fix it immediately and keep going until there are no more problems.
- Do not ask about ordering or sequencing — figure it out. If something is unclear, note it and skip it; only escalate when all paths are blocked.
- Obvious bugs: fix silently without reporting.
- No fallbacks or compatibility shims. One correct implementation per feature — no redundancy.

## Roadmap
- Public roadmap: `docs/roadmap.md` — tracks P0/P1 priorities, completed milestones, and timeline.
- Long-term vision: `docs/ultimate_goal.md` — the north star for where LEANN is headed.
- Keep in sync with [GitHub issue #237](https://github.com/yichuan-w/LEANN/issues/237).
- Welcome everyone to add more, and the craziest feature you want to put here! If people want some feature, all put there.

## Changelog (for contributors)
- Maintain `docs/CHANGELOG.md` — append-only log of major changes (new features, breaking changes, important fixes).
- Format: `## YYYY-MM-DD: <short summary>` followed by bullet points.
- Update the changelog when merging significant PRs or completing notable work.
- See `docs/CONTRIBUTING.md` for full contributor workflow (conventional commits, PR process, CI).

## Personal Dev Notes (gitignored)
- `docs/dev/` is gitignored for personal development notes (TODO, progress, experiments).
- Use `docs/dev/TODO.md` for in-progress tasks, `docs/dev/PROGRESS.md` for completed work.
- These are private scratch space — but must follow the Self-Contained Principle below.

## Documentation — Self-Contained Principle

All dev docs (`PROGRESS.md`, `STATES.md`, `EXPERIMENTS.md`, `TODO.md`) must be fully understandable from the document alone, with no reliance on conversation context or implied knowledge.

Requirements:
1. **Every technique/approach must be explained on first use.** Not "switched to IVF backend" — write "switched to IVF backend (FAISS IndexIVFFlat + DirectMap.Hashtable, supports in-place add/remove without full index rebuild)."
2. **Never assume the reader knows any abbreviation.** On first use: full name + one-sentence explanation. E.g., "HNSW (Hierarchical Navigable Small World — a graph-based ANN index used as LEANN's default backend)."
3. **Benchmark results must include full context.** Not "recall improved to 0.95" — write "recall@10 improved from 0.91 to 0.95 after switching from flat chunking (512 tokens, no overlap) to AST-aware chunking (function-level splits with 64-token overlap)."
4. **Numbers must have reference points.** Not "build time: 12s" — write "build time: 12s (vs. 45s before incremental update support, on 10k-document corpus)."
5. **Include the causal chain — not just conclusions.** Not "duplicate chunks appeared after incremental build" — write "Duplicate chunks appeared after incremental build because `passages.jsonl` was appended without first removing stale entries for modified files. The IVF index had correct vectors (remove-then-add), but the passage store was append-only, causing the same text to appear at multiple offsets."
6. **`docs/dev/STATES.md` top section maintains a glossary** of all key terms (backends, index files, chunking strategies, embedding models). Other docs reference it at the top.

Bad examples (forbidden):
- "Fixed the chunking bug" → Which bug? What was the symptom? What was the root cause?
- "Improved search quality" → By what metric? From what baseline? What change caused it?
- "Used nprobe=32" → What is nprobe? Why 32? What was it before and what effect did the change have?

## Doc Maintenance
- Maintain `docs/dev/PROGRESS.md` — completed work only (with key script/log/config paths). No plans.
- Maintain `docs/dev/TODO.md` — incomplete/in-progress/next-steps only (aim for one-command reproducibility). When done: remove from TODO, write result to PROGRESS, update STATES/EXPERIMENTS if needed.
- Both files: **append-only, chronological order** (oldest first). Use `tail -n 80 docs/dev/PROGRESS.md` to read recent entries; increase range or grep by date/keyword if needed.
- Keep TODO clean — either do items or remove them. Ask the user when unsure how to handle a TODO item.
- Maintain `docs/dev/STATES.md` — tracks all currently useful state (index configs, backend choices, known limitations); does NOT grow indefinitely (delete stale entries).
- Maintain `docs/dev/EXPERIMENTS.md` — benchmarks, A/B comparisons, parameter sweeps (recall@k, latency, storage size). Experimental content goes here, not in STATES.md.

## Commits
Commit when: (1) a complete feature is finished and tested, or (2) a destructive change is unavoidable.
```bash
git add <specific files>
git commit -m “feat: ...” # follow conventional commits
```
- When correcting errors: fix directly with no trace of the error.
- If you write a correct new version of a file, delete the wrong version. No duplicate implementations.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **LEANN-fork** (10372 symbols, 16907 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/LEANN-fork/context` | Codebase overview, check index freshness |
| `gitnexus://repo/LEANN-fork/clusters` | All functional areas |
| `gitnexus://repo/LEANN-fork/processes` | All execution flows |
| `gitnexus://repo/LEANN-fork/process/{name}` | Step-by-step execution trace |

## CLI

Public LEANN skills live under `skills/`. Local `.claude/skills/` and
`.agents/skills/` copies are ignored because they may contain personal runtime
settings.

| Task | Public skill file |
|------|-------------------|
| Build, rebuild, watch, or configure LEANN indexes | `skills/leann-index/SKILL.md` |
| Search LEANN indexes through CLI, MCP, or Python | `skills/leann-search/SKILL.md` |

<!-- gitnexus:end -->
