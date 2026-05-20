from __future__ import annotations

import json
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.manifest import SourceManifest
from leann_sources.readers import (
    APISourceReader,
    ExportZipSourceReader,
    FilesystemSourceReader,
    SQLiteSourceReader,
)


def _assert_utc_iso(value: str) -> None:
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def _assert_source_axes(metadata: dict) -> None:
    for axis in ("created_at", "modified_at", "event_time"):
        assert axis in metadata
        _assert_utc_iso(metadata[axis])
    assert "indexed_at" not in metadata


def _manifest(data: dict, fields: dict, *, source_type: str = "document") -> SourceManifest:
    return SourceManifest.from_dict(
        {
            "name": source_type,
            "category": "test",
            "display_name": source_type.title(),
            "version": "0.1.0",
            "manifest_version": "1.0",
            "data": data,
            "auth": {"type": "none"},
            "fields": {
                **fields,
                "source_type": {"value": source_type},
            },
            "chunking": {
                "granularity": "message",
                "text_template": "{body}",
                "max_chunk_tokens": 400,
            },
            "privacy": {"tier": "tier_1"},
        }
    )


def test_sqlite_reader_emits_signals_chunks(tmp_path: Path):
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "create table messages (id integer, body text, sent_at integer, edited_at integer, sender text, participants text)"
        )
        connection.execute(
            "insert into messages values (7, 'hello https://example.com', 0, 60, 'alice', 'alice,bob')"
        )
        connection.execute(
            "insert into messages values (8, 'second message', 30, 90, 'alice', 'alice,bob')"
        )

    manifest = _manifest(
        {
            "type": "sqlite",
            "default_path": str(db_path),
            "query": "select * from messages order by id",
            "count_query": "select count(*) from messages",
        },
        {
            "event_time": {"source": "sent_at", "transform": "unix_to_utc_iso", "required": True},
            "created_at": {
                "source": "sent_at",
                "transform": "unix_to_utc_iso",
                "synthesized_from": "event_time",
            },
            "modified_at": {"source": "edited_at", "transform": "unix_to_utc_iso"},
            "author": {"source": "sender"},
            "participant_ids": {"source": "participants", "transform": "list_of_handles"},
            "source_id": {"source": "id", "transform": "format('msg_{value}')"},
            "source_document_id": {"source": "id", "transform": "format('thread_{value}')"},
            "chunk_seq": {"auto": "row_index_within_source_document_id"},
            "mentioned_urls": {"extraction": "regex", "pattern": r"https?://[^\s>]+"},
        },
        source_type="imessage",
    )

    reader = SQLiteSourceReader(manifest)
    chunks = list(reader.iter_chunks())

    assert reader.discover().found
    assert reader.validate().ok
    assert reader.stats().count == 2
    assert chunks[0].text == "hello https://example.com"
    assert chunks[0].metadata["source_id"] == "msg_7"
    assert chunks[0].metadata["chunk_seq"] == 0
    assert chunks[1].metadata["chunk_seq"] == 1
    assert chunks[0].metadata["created_at_synthesized"] is True
    assert chunks[0].metadata["participant_ids"] == ["alice", "bob"]
    assert chunks[0].metadata["mentioned_urls"] == ["https://example.com"]
    _assert_source_axes(chunks[0].metadata)


def test_filesystem_reader_uses_birthtime_and_mtime(tmp_path: Path):
    source_file = tmp_path / "note.txt"
    source_file.write_text("filesystem note", encoding="utf-8")
    stat = source_file.stat()
    if not hasattr(stat, "st_birthtime"):
        pytest.skip("st_birthtime is macOS-specific")

    manifest = _manifest(
        {"type": "filesystem", "default_path": str(tmp_path), "glob": "*.txt"},
        {
            "created_at": {"source": "stat.st_birthtime", "transform": "unix_to_utc_iso"},
            "modified_at": {"source": "stat.st_mtime", "transform": "unix_to_utc_iso"},
            "event_time": {
                "source": "stat.st_birthtime",
                "transform": "unix_to_utc_iso",
                "synthesized_from": "created_at",
            },
            "source_id": {"source": "relative_path"},
            "source_document_id": {"source": "relative_path"},
            "chunk_seq": {"auto": "row_index_within_source_document_id"},
        },
    )

    chunk = next(FilesystemSourceReader(manifest).iter_chunks())

    assert chunk.text == "filesystem note"
    assert chunk.metadata["source_id"] == "note.txt"
    assert chunk.metadata["event_time_synthesized"] is True
    _assert_source_axes(chunk.metadata)


def test_api_reader_paginates_fixture_pages_and_tracks_validate():
    manifest = _manifest(
        {
            "type": "api",
            "base_url": "https://api.example.test",
            "pages": [
                [
                    {
                        "id": 1,
                        "body": "api body",
                        "created": 0,
                        "updated": 60,
                    }
                ]
            ],
        },
        {
            "created_at": {"source": "created", "transform": "unix_to_utc_iso"},
            "modified_at": {"source": "updated", "transform": "unix_to_utc_iso"},
            "event_time": {
                "source": "created",
                "transform": "unix_to_utc_iso",
                "synthesized_from": "created_at",
            },
            "source_id": {"source": "id", "transform": "format('api_{value}')"},
            "source_document_id": {"source": "id", "transform": "format('api_doc_{value}')"},
            "chunk_seq": {"auto": "row_index"},
        },
        source_type="github",
    )

    reader = APISourceReader(manifest)
    chunk = next(reader.iter_chunks())

    assert reader.discover().found
    assert reader.validate().ok
    assert reader.stats().count == 1
    assert chunk.metadata["source_id"] == "api_1"
    assert chunk.metadata["event_time_synthesized"] is True
    _assert_source_axes(chunk.metadata)


def test_export_zip_reader_emits_json_records(tmp_path: Path):
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "messages.json",
            json.dumps(
                [
                    {
                        "id": "a",
                        "body": "zip body",
                        "created": 0,
                        "updated": 60,
                    }
                ]
            ),
        )

    manifest = _manifest(
        {"type": "export_zip", "default_path": str(zip_path)},
        {
            "created_at": {"source": "created", "transform": "unix_to_utc_iso"},
            "modified_at": {"source": "updated", "transform": "unix_to_utc_iso"},
            "event_time": {
                "source": "created",
                "transform": "unix_to_utc_iso",
                "synthesized_from": "created_at",
            },
            "source_id": {"source": "id", "transform": "format('zip_{value}')"},
            "source_document_id": {"source": "source_document_id"},
            "chunk_seq": {"auto": "row_index_within_source_document_id"},
        },
        source_type="chatgpt-export",
    )

    reader = ExportZipSourceReader(manifest)
    chunk = next(reader.iter_chunks())

    assert reader.discover().found
    assert reader.validate().ok
    assert reader.stats().count == 1
    assert chunk.text == "zip body"
    assert chunk.metadata["source_id"] == "zip_a"
    assert chunk.metadata["source_document_id"] == "messages.json"
    _assert_source_axes(chunk.metadata)
