# CLI Reference

Comprehensive list of LEANN commands and options.

## Global Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LEANN_HOME` | Directory where all indexes are stored | `~/.leann` |
| `OLLAMA_HOST`| Host address for the Ollama API | `http://localhost:11434` |

## Primary Commands

### `build`
Create a new index from local files.
- `--docs`: Path to directory or file.
- `--backend-name`: `hnsw` or `diskann`.
- `--use-ast-chunking`: Enable code-aware parsing.
- `--doc-chunk-size`: Token limit per chunk (default: 256).

### `search`
Fast semantic retrieval.
- `--top-k`: Number of results to return.
- `--no-recompute`: Faster search using cached embeddings (if available).
- `--complexity`: Search depth (higher = more accurate, slower).

### `ask`
Interactive RAG session.
- `--model`: LLM model name (e.g., `gpt-oss:20b`).
- `--interactive`: Start a persistent chat session.

## Source-Specific Commands

| Command | Target Source |
| :--- | :--- |
| `index-email` | Apple Mail (local SQLite) |
| `index-calendar`| Apple Calendar events |
| `index-imessage`| iMessage history |
| `index-browser` | Chrome/Brave history |
| `index-chatgpt` | ChatGPT JSON/ZIP exports |
| `index-claude` | Claude JSON/ZIP exports |

## Management Commands

### `list`
Instantly list all indexes across all registered projects. Highly optimized for large storage systems.

### `remove`
Safely delete an index.
- `-f`, `--force`: Bypass confirmation (only if no naming ambiguity exists).
