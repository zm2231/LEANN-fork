#!/usr/bin/env python3
"""Build a unified ``coding-sessions`` LEANN index from the 3 agent-session readers.

Usage:
    LEANN_LOCAL_USER=zain \\
    LEANN_PI_AGENT_ROOTS=/Volumes/4/GitHub/pi-ult/sessions \\
    .venv/bin/python tools/build_coding_sessions.py [--max-count N] [--index NAME]

The combined index carries `source_type=claude_code|codex|pi_agent` and `project_id=...`
in every chunk's metadata so the index can be filtered at search time:

    leann search coding-sessions "QUERY" --filter source_type=codex,project_id=studymill
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCES_ROOT = ROOT / "packages" / "leann-sources" / "sources"
sys.path.insert(0, str(ROOT / "packages" / "leann-sources" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "leann-core" / "src"))

from leann_sources.manifest import SourceManifest  # noqa: E402

AGENT_SESSIONS_ROOT = SOURCES_ROOT / "agent-sessions"

READER_SPECS = [
    ("claude-code", "ClaudeCodeSourceReader"),
    ("codex", "CodexSourceReader"),
    ("pi-agent", "PiAgentSourceReader"),
]


def _load_reader(name: str, cls_name: str):
    reader_path = AGENT_SESSIONS_ROOT / name / "reader.py"
    spec = importlib.util.spec_from_file_location(f"_build_reader_{name}", reader_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = SourceManifest.load(AGENT_SESSIONS_ROOT / name / "manifest.yaml")
    return getattr(module, cls_name)(manifest)


def _cwd_match(cwd: str | None, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    if not cwd:
        return False
    return any(cwd == p or cwd.startswith(p + "/") for p in prefixes)


def _iter_chunks(
    max_count_per_source: int,
    since: datetime | None = None,
    cwd_prefixes: tuple[str, ...] = (),
):
    for name, cls_name in READER_SPECS:
        reader = _load_reader(name, cls_name)
        report = reader.validate()
        if not report.ok:
            print(f"[{name}] skipped: {report.status} ({report.errors})", file=sys.stderr)
            continue
        count = 0
        for chunk in reader.iter_chunks(since=since):
            if cwd_prefixes and not _cwd_match(
                chunk.metadata.get("extra", {}).get("cwd"), cwd_prefixes
            ):
                continue
            yield name, chunk
            count += 1
            if max_count_per_source > 0 and count >= max_count_per_source:
                break
        print(f"[{name}] emitted {count} chunks", file=sys.stderr)


async def main(args: argparse.Namespace) -> None:
    from leann.cli import LeannCLI
    from llama_index.core import Document

    documents: list[Document] = []
    per_source_counts: dict[str, int] = {}
    long_chunk_count = 0
    char_cap = args.chunk_char_cap
    since = None
    if args.since_days > 0:
        since = datetime.now(UTC) - timedelta(days=args.since_days)
        print(f"[build] since={since.isoformat()} ({args.since_days}d window)", file=sys.stderr)
    cwd_prefixes = tuple(p for p in (args.cwd_prefix or []) if p)
    if cwd_prefixes:
        print(f"[build] cwd-prefix filter: {cwd_prefixes}", file=sys.stderr)
    for source_name, chunk in _iter_chunks(args.max_count, since=since, cwd_prefixes=cwd_prefixes):
        text = chunk.text
        if char_cap > 0 and len(text) > char_cap:
            text = text[:char_cap] + f"\n…[truncated to {char_cap} chars]"
            long_chunk_count += 1
        documents.append(Document(text=text, metadata=chunk.metadata))
        per_source_counts[source_name] = per_source_counts.get(source_name, 0) + 1

    if long_chunk_count:
        print(
            f"\n[build] truncated {long_chunk_count} oversized chunks to {char_cap} chars",
            file=sys.stderr,
        )

    print(f"\nTotal documents: {len(documents)}", file=sys.stderr)
    print(f"By source: {per_source_counts}", file=sys.stderr)

    cli = LeannCLI()
    cli.register_project_dir()
    build_args = SimpleNamespace(
        index_name=args.index,
        embedding_model=args.embedding_model,
        embedding_mode=args.embedding_mode,
        embedding_host=args.embedding_host,
        embedding_api_base=args.embedding_api_base,
        embedding_api_key=args.embedding_api_key,
        no_recompute=True,
        backend_name=args.backend,
        force=args.force,
    )
    await cli._build_index_from_documents(build_args, documents)
    print(f"\nBuilt index: {args.index}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the coding-sessions LEANN index")
    parser.add_argument("--index", default="coding-sessions", help="Target index name")
    parser.add_argument(
        "--max-count",
        type=int,
        default=-1,
        help="Per-source chunk cap (-1 = unlimited, default)",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--embedding-mode",
        default="openai",
        choices=["sentence-transformers", "openai", "mlx", "ollama"],
    )
    parser.add_argument("--embedding-host", default=None)
    parser.add_argument(
        "--embedding-api-base",
        default="http://100.122.112.83:8100/v1",
        help="OpenAI-compatible embedding base (default: iq endpoint)",
    )
    parser.add_argument("--embedding-api-key", default="ignored")
    parser.add_argument("--backend", default="hnsw")
    parser.add_argument("--force", action="store_true", help="Overwrite existing index")
    parser.add_argument(
        "--cwd-prefix",
        action="append",
        default=None,
        help="Only ingest chunks whose session cwd matches this path prefix (repeatable)",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Only ingest sessions with mtime within the last N days (0 = no window, default)",
    )
    parser.add_argument(
        "--chunk-char-cap",
        type=int,
        default=8000,
        help="Truncate chunk text to this many chars before indexing (0 = no cap, default 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
