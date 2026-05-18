#!/usr/bin/env python3
import argparse
import glob
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from leann.api import LeannBuilder

DB_PATH = Path("/Users/zain/Documents/leann-eval-data/slacrawl.db")
OUT_DIR = Path("/Users/zain/Documents/leann-eval-data/metadata-aware")
SPARSE_INDEX = OUT_DIR / "eval-sparse-corpus"
SLACK_INDEX = OUT_DIR / "eval-slack"
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_BASE_URL = "http://localhost:8100/v1"
URL_RE = re.compile(r"https?://[^\s<>\"]+")

SPARSE_CHANNELS = {
    "big-brain",
    "btd-community",
    "content-accountability",
    "dev-videos",
    "jai-newsletter",
    "leads",
    "obsidian-buddy",
    "partnerships",
    "product-strategy",
    "resources-prompts-tools-skills",
    "team-shoutouts",
    "test-automations",
    "wow-channel",
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _cleanup_index(index_path: Path) -> None:
    for path in glob.glob(f"{index_path}*"):
        candidate = Path(path)
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()


def _message_rows(conn: sqlite3.Connection, sparse: bool) -> list[sqlite3.Row]:
    clauses = ["trim(m.text) != ''", "(m.deleted_ts is null or m.deleted_ts = '')"]
    params: list[Any] = []
    if sparse:
        placeholders = ",".join("?" for _ in SPARSE_CHANNELS)
        clauses.append(f"(c.name in ({placeholders}) or m.text like '%loom.com%')")
        params.extend(sorted(SPARSE_CHANNELS))

    query = f"""
        select
            m.channel_id,
            c.name as channel_name,
            m.ts,
            m.thread_ts,
            m.user_id,
            m.text,
            m.normalized_text
        from messages m
        join channels c on c.id = m.channel_id
        where {" and ".join(clauses)}
        order by m.channel_id, cast(m.ts as real)
    """
    return list(conn.execute(query, params))


def _metadata_for_row(row: sqlite3.Row, chunk_seq: int) -> dict[str, Any]:
    text = row["normalized_text"] or row["text"]
    msg_ts = row["ts"]
    thread_or_msg_ts = row["thread_ts"] or msg_ts
    channel_id = row["channel_id"]
    channel_name = row["channel_name"] or channel_id
    return {
        "id": f"{channel_id}:{msg_ts}",
        "source_type": "slack",
        "source_id": msg_ts,
        "source_document_id": f"{channel_id}:{thread_or_msg_ts}",
        "chunk_seq": chunk_seq,
        "author": row["user_id"] or "",
        "parent_ref": f"channel:{channel_name}",
        "mentioned_urls": URL_RE.findall(text),
    }


def _build_index(index_path: Path, rows: list[sqlite3.Row], force: bool) -> None:
    meta_path = Path(f"{index_path}.meta.json")
    if meta_path.exists() and not force:
        print(f"{index_path.name}: exists, skipping ({meta_path})")
        return

    _cleanup_index(index_path)
    builder = LeannBuilder(
        backend_name="hnsw",
        embedding_model=EMBEDDING_MODEL,
        embedding_mode="openai",
        embedding_options={
            "base_url": EMBEDDING_BASE_URL,
            "api_key": "local-iq",
            "batch_size": 64,
        },
        is_recompute=False,
        is_compact=False,
        M=16,
        efConstruction=200,
    )

    seq_by_channel: dict[str, int] = {}
    added = 0
    for row in rows:
        channel_id = row["channel_id"]
        chunk_seq = seq_by_channel.get(channel_id, 0)
        seq_by_channel[channel_id] = chunk_seq + 1
        text = row["normalized_text"] or row["text"]
        metadata = _metadata_for_row(row, chunk_seq)
        builder.add_text(text, metadata=metadata)
        added += 1

    print(f"{index_path.name}: building {added} passages")
    builder.build_index(str(index_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild existing eval indexes")
    parser.add_argument(
        "--only",
        choices=("all", "sparse", "slack"),
        default="all",
        help="Build only one eval index",
    )
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "local-iq")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        if args.only in {"all", "sparse"}:
            _build_index(SPARSE_INDEX, _message_rows(conn, sparse=True), args.force)
        if args.only in {"all", "slack"}:
            _build_index(SLACK_INDEX, _message_rows(conn, sparse=False), args.force)


if __name__ == "__main__":
    main()
