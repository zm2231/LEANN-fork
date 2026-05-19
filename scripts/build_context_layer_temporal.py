#!/usr/bin/env python3
"""Build the Wave 1.5 context-layer temporal eval index."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from leann.api import LeannBuilder
from leann.cli import LeannCLI
from leann.embedding_compute import compute_embeddings as compute_embeddings_direct
from llama_index.core import Document

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "100.70.176.74:/Volumes/4/GitHub/context-layer"
DEFAULT_STAGING_DIR = Path("/private/tmp/context-layer-temporal-src")
INDEX_NAME = "context-layer-temporal"
GOLD_PATH = ROOT / "tests" / "eval" / "context_layer_temporal_gold.jsonl"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BASE_URL = "http://100.122.112.83:8100/v1"
TEMPORAL_NOW = "2026-05-19T12:00:00+00:00"
TEXT_EXTENSIONS = {".md", ".txt", ".py", ".json", ".ts", ".tsx", ".sql", ".yml", ".yaml"}
TEMPORAL_AXES = ["created_at", "modified_at", "event_time", "indexed_at"]
EXCLUDED_PARTS = {
    ".git",
    ".leann",
    ".zen/logs",
    ".claude/worktrees",
    "postgres-data",
    "qdrant-storage",
    "redis-data",
    "node_modules",
    ".next",
    "dist",
    "build",
}


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + max_chars, len(cleaned))
        if end < len(cleaned):
            split_at = cleaned.rfind("\n\n", start, end)
            if split_at > start + max_chars // 2:
                end = split_at
        chunks.append(cleaned[start:end].strip())
        if end == len(cleaned):
            break
        start = max(end - overlap, 0)
    return [chunk for chunk in chunks if chunk]


def is_remote_source(source: str) -> bool:
    return ":" in source and not Path(source).exists()


def sync_remote_source(source: str, staging_dir: Path) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    remote = source if source.endswith("/") else f"{source}/"
    excludes = [
        ".git/",
        ".leann/",
        ".zen/logs/",
        ".claude/worktrees/",
        "postgres-data/",
        "qdrant-storage/",
        "redis-data/",
        "node_modules/",
        ".next/",
        "dist/",
        "build/",
    ]
    command = ["rsync", "-az", "--delete", "--delete-excluded"]
    for exclude in excludes:
        command.extend(["--exclude", exclude])
    command.extend([remote, f"{staging_dir}/"])
    subprocess.run(command, check=True)
    return staging_dir


def resolve_source_root(source: str, staging_dir: Path) -> Path:
    if is_remote_source(source):
        return sync_remote_source(source, staging_dir)
    return Path(source).expanduser().resolve()


def should_skip(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    parts = set(relative.parts)
    if parts & {
        ".git",
        ".leann",
        "postgres-data",
        "qdrant-storage",
        "redis-data",
        "node_modules",
        ".next",
        "dist",
        "build",
    }:
        return True
    rel_text = relative.as_posix()
    return any(rel_text.startswith(prefix) for prefix in EXCLUDED_PARTS if "/" in prefix)


def iter_source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if should_skip(path, root):
            continue
        yield path


def file_temporal_metadata(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    metadata: dict[str, Any] = {
        "modified_at": utc_iso(datetime.fromtimestamp(stat_result.st_mtime, timezone.utc))
    }
    birthtime = getattr(stat_result, "st_birthtime", None)
    if birthtime is not None:
        metadata["created_at"] = utc_iso(datetime.fromtimestamp(birthtime, timezone.utc))
    return metadata


def build_documents(source_root: Path) -> list[Document]:
    documents: list[Document] = []
    for path in iter_source_files(source_root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = path.relative_to(source_root).as_posix()
        temporal = file_temporal_metadata(path)
        for chunk_index, chunk in enumerate(chunk_text(text)):
            source_id = f"{relative}#chunk-{chunk_index}"
            documents.append(
                Document(
                    text=chunk,
                    metadata={
                        "source_type": "context_layer",
                        "source_id": source_id,
                        "source_url": path.as_uri(),
                        "project_id": "context-layer",
                        "parent_ref": f"file:{relative}",
                        "mentioned_files": re.findall(r"[\w./-]+\.(?:ts|tsx|py|sql|md|json)", chunk),
                        "mentioned_refs": re.findall(r"(?:#\d+|WAVE-\d+|PHASE-\d+)", chunk),
                        "mentioned_urls": re.findall(r"https?://[^\s)>\"]+", chunk),
                        "extra": {"file_name": path.name, "chunk_index": chunk_index},
                        **temporal,
                    },
                )
            )
    return documents


def validate_documents(documents: list[Document]) -> None:
    if not documents:
        raise SystemExit("No context-layer documents collected")
    required = {"source_type", "source_id", "created_at", "modified_at"}
    for doc in documents:
        missing = required - set(doc.metadata)
        if missing:
            raise SystemExit(f"Document {doc.metadata.get('source_id')} missing {sorted(missing)}")


def gold_rows(documents: list[Document]) -> list[dict[str, Any]]:
    ids = [doc.metadata["source_id"] for doc in documents]
    selected = ids[:50]
    if len(selected) < 50:
        raise SystemExit(f"Need at least 50 chunks for gold, found {len(selected)}")

    buckets = (
        [("created_at", "created", "non_adversarial")] * 10
        + [("modified_at", "edited", "non_adversarial")] * 10
        + [("event_time", "around", "non_adversarial")] * 10
        + [("event_time", "recent", "non_adversarial")] * 10
        + [("created_at", "on", "adversarial")] * 5
        + [("modified_at", "early", "adversarial")] * 5
    )
    rows = []
    for index, (axis, verb, bucket) in enumerate(buckets):
        source_id = selected[index]
        parent = source_id.split("#chunk-", 1)[0]
        rows.append(
            {
                "query": f"{parent} {verb} in May",
                "axis_expected": axis,
                "window_expected": ["2026-05-01T00:00:00+00:00", "2026-05-31T23:59:59+00:00"],
                "gold_ids": [source_id],
                "bucket": bucket,
                "now": TEMPORAL_NOW,
            }
        )
    return rows


def write_gold(documents: list[Document], output_path: Path = GOLD_PATH) -> None:
    rows = gold_rows(documents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def validate_gold(path: Path = GOLD_PATH) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 50:
        raise SystemExit(f"Expected 50 gold rows, found {len(rows)}")
    allowed_axes = {"created_at", "modified_at", "event_time", "indexed_at"}
    non_adversarial = 0
    for row in rows:
        if row["axis_expected"] not in allowed_axes:
            raise SystemExit(f"Invalid axis in gold row: {row}")
        if len(row["window_expected"]) != 2:
            raise SystemExit(f"Invalid window in gold row: {row}")
        if not row["gold_ids"]:
            raise SystemExit(f"Missing gold_ids in gold row: {row}")
        if row.get("bucket") != "adversarial":
            non_adversarial += 1
    if non_adversarial != 40:
        raise SystemExit(f"Expected 40 non-adversarial rows, found {non_adversarial}")


async def build_index(documents: list[Document]) -> None:
    cli = LeannCLI()
    index_path = cli.get_index_path(INDEX_NAME)
    shutil.rmtree(cli.indexes_dir / INDEX_NAME, ignore_errors=True)
    embedding_options = {"base_url": EMBEDDING_BASE_URL, "api_key": "iq-local"}
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=EMBEDDING_MODEL,
        embedding_mode="openai",
        embedding_options=embedding_options,
        is_recompute=False,
    )
    indexed_at = datetime.now(timezone.utc).isoformat()
    for doc in documents:
        builder.add_text(doc.text, metadata={**doc.metadata, "indexed_at": indexed_at})

    texts = [chunk["text"] for chunk in builder.chunks]
    embeddings = [
        compute_embeddings_direct(
            texts[start : start + 128],
            EMBEDDING_MODEL,
            mode="openai",
            is_build=True,
            provider_options=embedding_options,
        )
        for start in range(0, len(texts), 128)
    ]
    builder.build_index_from_arrays(
        index_path,
        [chunk["id"] for chunk in builder.chunks],
        np.vstack(embeddings),
    )
    annotate_index_meta(index_path)
    validate_built_index(index_path)
    cli.register_project_dir()
    print(f"Index '{INDEX_NAME}' built at {index_path}")


def annotate_index_meta(index_path: str) -> None:
    meta_path = Path(f"{index_path}.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["metadata_temporal_axes"] = TEMPORAL_AXES
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def validate_built_index(index_path: str) -> None:
    index = Path(index_path)
    meta_path = Path(f"{index_path}.meta.json")
    passages_path = index.parent / f"{index.name}.passages.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("metadata_temporal_axes") != TEMPORAL_AXES:
        raise SystemExit(f"{meta_path} missing metadata_temporal_axes")
    rows_checked = 0
    for line in passages_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        metadata = json.loads(line)["metadata"]
        missing = {"created_at", "modified_at", "indexed_at"} - set(metadata)
        if missing:
            raise SystemExit(f"Passage missing temporal metadata: {sorted(missing)}")
        if "event_time" in metadata:
            raise SystemExit("Filesystem context-layer passage unexpectedly includes event_time")
        rows_checked += 1
    if rows_checked == 0:
        raise SystemExit(f"No passages found in {passages_path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=DEFAULT_REMOTE)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--gold-only", action="store_true")
    parser.add_argument("--validate-gold-only", action="store_true")
    args = parser.parse_args()

    if args.validate_gold_only:
        validate_gold()
        return

    source_root = resolve_source_root(args.source_root, args.staging_dir)
    documents = build_documents(source_root)
    validate_documents(documents)
    write_gold(documents)
    validate_gold()
    print(f"Collected {len(documents)} context-layer chunks from {source_root}")
    if not args.gold_only:
        await build_index(documents)


if __name__ == "__main__":
    asyncio.run(main())
