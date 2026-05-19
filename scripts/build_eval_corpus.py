#!/usr/bin/env python3
"""Build the Wave 1 temporal eval indexes."""

from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import sqlite3
import subprocess
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from leann.api import LeannBuilder
from leann.cli import LeannCLI
from leann.embedding_compute import compute_embeddings as compute_embeddings_direct
from llama_index.core import Document

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path("/Users/zain/Documents/jay-abraham-eval/Jay-Abraham-Curated")
SLACK_DB = Path("/Users/zain/Documents/leann-eval-data/slacrawl.db")
SUMMARY_DIR = Path("/Users/zain/Documents/leann-eval-data/notion")
INDEX_NAMES = {
    "document": "eval-docs",
    "git_commit": "eval-commits",
    "slack": "eval-slack",
    "daily_summary": "eval-summaries",
}
EMBEDDING_BATCH_SIZE = 128
SLACK_CHUNK_CHARS = 800
SLACK_CHUNK_OVERLAP = 80
URL_RE = re.compile(r"<(https?://[^>|]+)")
REF_RE = re.compile(r"(#\w+|<@U\w+>)")


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def mentioned_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def mentioned_refs(text: str) -> list[str]:
    return REF_RE.findall(text or "")


def index_meta_path(cli: LeannCLI, index_name: str) -> Path:
    return Path(cli.get_index_path(index_name) + ".meta.json")


def should_rebuild(cli: LeannCLI, index_name: str, sources: Iterable[Path], force: bool) -> bool:
    if force:
        return True
    meta_path = index_meta_path(cli, index_name)
    if not meta_path.exists():
        return True
    index_mtime = meta_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > index_mtime for path in sources)


async def build_documents(cli: LeannCLI, index_name: str, documents: list[Document]) -> None:
    embedding_options = {
        "base_url": "http://localhost:8100/v1",
        "api_key": "iq-local",
    }
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model="BAAI/bge-m3",
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
            texts[start : start + EMBEDDING_BATCH_SIZE],
            "BAAI/bge-m3",
            mode="openai",
            is_build=True,
            provider_options=embedding_options,
        )
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE)
    ]
    index_path = cli.get_index_path(index_name)
    builder.build_index_from_arrays(
        index_path,
        [chunk["id"] for chunk in builder.chunks],
        np.vstack(embeddings),
    )
    cli.register_project_dir()
    print(f"Index '{index_name}' built at {index_path}")


def clean_index(cli: LeannCLI, index_name: str) -> None:
    shutil.rmtree(cli.indexes_dir / index_name, ignore_errors=True)


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_pdf_text(path: Path) -> str:
    import fitz

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


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


def ingest_docs() -> list[Document]:
    docs: list[Document] = []
    for stat_path in sorted(path for path in DOCS_DIR.iterdir() if path.is_file()):
        suffix = stat_path.suffix.lower()
        if suffix == ".md":
            text = stat_path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            text = extract_docx_text(stat_path)
        elif suffix == ".pdf":
            text = extract_pdf_text(stat_path)
        else:
            continue
        stat_result = stat_path.stat()
        created_at = utc_iso(datetime.fromtimestamp(stat_result.st_birthtime, timezone.utc))
        modified_at = utc_iso(datetime.fromtimestamp(stat_result.st_mtime, timezone.utc))
        for chunk_index, chunk in enumerate(chunk_text(text)):
            docs.append(
                Document(
                    text=chunk,
                    metadata={
                        "source_type": "document",
                        "source_id": f"{stat_path}#chunk-{chunk_index}",
                        "source_url": stat_path.as_uri() if stat_path.exists() else None,
                        "created_at": created_at,
                        "modified_at": modified_at,
                        "project_id": "jay-abraham-eval",
                        "parent_ref": f"file:{stat_path.name}",
                        "mentioned_urls": mentioned_urls(chunk),
                        "mentioned_refs": mentioned_refs(chunk),
                        "mentioned_files": [],
                        "extra": {"file_name": stat_path.name, "chunk_index": chunk_index},
                    },
                )
            )
    return docs


def ingest_commits() -> list[Document]:
    fmt = "%H%x1f%aI%x1f%cI%x1f%an%x1f%s%x1f%b%x1e"
    raw = subprocess.check_output(
        ["git", "log", "--all", "--no-merges", "--since=2025-06-01", f"--format={fmt}"],
        cwd=ROOT,
        text=True,
    )
    docs: list[Document] = []
    for record in raw.strip("\x1e\n").split("\x1e"):
        if not record.strip():
            continue
        sha, authored_at, committed_at, author, subject, body = record.strip("\n").split("\x1f", 5)
        files = subprocess.check_output(
            ["git", "show", "--pretty=", "--name-only", sha],
            cwd=ROOT,
            text=True,
        ).splitlines()
        embedded_files = files[:20]
        body_excerpt = body[:1000]
        event_time = utc_iso(datetime.fromisoformat(authored_at))
        modified_at = utc_iso(datetime.fromisoformat(committed_at))
        text = (
            f"Commit {sha}\nAuthor: {author}\nDate: {event_time}\nSubject: {subject}\n"
            f"{body_excerpt}\nFiles:\n" + "\n".join(embedded_files)
        )
        docs.append(
            Document(
                text=text,
                metadata={
                    "source_type": "git_commit",
                    "source_id": sha,
                    "source_url": f"https://github.com/elyxlz/LEANN/commit/{sha}",
                    "created_at": event_time,
                    "modified_at": modified_at,
                    "event_time": event_time,
                    "author": author,
                    "activity_type": "authored",
                    "participant_ids": [author],
                    "project_id": "leann",
                    "parent_ref": "repo:LEANN",
                    "mentioned_urls": mentioned_urls(text),
                    "mentioned_refs": re.findall(r"(#\d+|[A-Z]+-\d+)", text),
                    "mentioned_files": embedded_files,
                    "extra": {"file_count": len(files)},
                },
            )
        )
    return docs


def ingest_slack() -> list[Document]:
    conn = sqlite3.connect(SLACK_DB)
    rows = conn.execute(
        """
        SELECT m.channel_id, COALESCE(c.name, m.channel_id), m.ts, m.user_id,
               COALESCE(m.thread_ts, ''), m.text
        FROM messages m LEFT JOIN channels c ON c.id = m.channel_id
        WHERE m.ts NOT LIKE 'draft:%' AND (m.deleted_ts IS NULL OR m.deleted_ts = '')
        ORDER BY CAST(m.ts AS REAL)
        """
    ).fetchall()
    conn.close()
    docs: list[Document] = []
    for channel_id, channel_name, ts, user_id, thread_ts, text in rows:
        event_time = utc_iso(datetime.fromtimestamp(float(ts), timezone.utc))
        body = f"Slack #{channel_name} {event_time} {user_id or ''}\n{text}"
        chunks = chunk_text(body, max_chars=SLACK_CHUNK_CHARS, overlap=SLACK_CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    text=chunk,
                    metadata={
                        "source_type": "slack",
                        "source_id": f"{channel_id}:{ts}:chunk:{idx}",
                        "source_url": f"slack://channel/{channel_id}/p{ts.replace('.', '')}",
                        "created_at": event_time,
                        "event_time": event_time,
                        "author": user_id,
                        "activity_type": "authored" if user_id else None,
                        "participant_ids": [user_id] if user_id else [],
                        "parent_ref": f"channel:{channel_name}",
                        "mentioned_urls": mentioned_urls(text),
                        "mentioned_refs": mentioned_refs(text),
                        "mentioned_files": [],
                        "extra": {
                            "thread_ts": thread_ts or ts,
                            "channel_id": channel_id,
                            "chunk_index": idx,
                            "chunk_count": len(chunks),
                        },
                    },
                )
            )
    return docs


def ingest_summaries() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(SUMMARY_DIR.glob("*/*.md")):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", path.name):
            continue
        event_date = datetime.fromisoformat(path.stem).replace(tzinfo=timezone.utc)
        text = path.read_text(encoding="utf-8")
        channel = path.parent.name
        for chunk_index, chunk in enumerate(chunk_text(text)):
            docs.append(
                Document(
                    text=chunk,
                    metadata={
                        "source_type": "daily_summary",
                        "source_id": f"{path.relative_to(SUMMARY_DIR)}#chunk-{chunk_index}",
                        "event_time": utc_iso(event_date),
                        "author": None,
                        "participant_ids": [],
                        "parent_ref": f"channel:{channel}",
                        "mentioned_urls": mentioned_urls(chunk),
                        "mentioned_refs": mentioned_refs(chunk),
                        "mentioned_files": [],
                        "extra": {"file_name": path.name, "chunk_index": chunk_index},
                    },
                )
            )
    return docs


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild all indexes")
    parser.add_argument(
        "--only",
        choices=sorted(INDEX_NAMES),
        action="append",
        help="Limit rebuild to one source type; can be repeated",
    )
    args = parser.parse_args()

    cli = LeannCLI()
    plans = [
        (INDEX_NAMES["document"], ingest_docs, [DOCS_DIR, Path(__file__)]),
        (INDEX_NAMES["git_commit"], ingest_commits, [ROOT / ".git" / "HEAD", Path(__file__)]),
        (INDEX_NAMES["slack"], ingest_slack, [SLACK_DB, Path(__file__)]),
        (INDEX_NAMES["daily_summary"], ingest_summaries, [SUMMARY_DIR, Path(__file__)]),
    ]
    only = set(args.only or [])
    for index_name, ingest, sources in plans:
        source_type = next(key for key, value in INDEX_NAMES.items() if value == index_name)
        if only and source_type not in only:
            print(f"Skipping {index_name}: not selected by --only.")
            continue
        if not should_rebuild(cli, index_name, sources, args.force):
            print(f"Index '{index_name}' is current; skipping.")
            continue
        clean_index(cli, index_name)
        documents = ingest()
        print(f"Building {index_name}: {len(documents)} chunks")
        await build_documents(cli, index_name, documents)


if __name__ == "__main__":
    asyncio.run(main())
