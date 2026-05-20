from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest


ROOT = Path(__file__).resolve().parents[1]
IMESSAGE_MANIFEST = ROOT / "sources" / "messaging" / "imessage" / "manifest.yaml"


def _cocoa_ns(value: datetime) -> int:
    cocoa_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return int((value.timestamp() - cocoa_epoch.timestamp()) * 1_000_000_000)


def _write_chat_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY,
            text TEXT,
            date INTEGER,
            date_edited INTEGER,
            is_from_me INTEGER,
            service TEXT,
            handle_id INTEGER
        );
        CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        """
    )
    created = datetime(2026, 5, 12, 10, tzinfo=timezone.utc)
    edited = datetime(2026, 5, 12, 11, tzinfo=timezone.utc)
    connection.execute("INSERT INTO chat VALUES (1, 'chat-1', 'Chat One')")
    connection.execute("INSERT INTO handle VALUES (1, '+15555550123')")
    connection.execute(
        "INSERT INTO message VALUES (1, 'hello https://example.com', ?, ?, 0, 'iMessage', 1)",
        (_cocoa_ns(created), _cocoa_ns(edited)),
    )
    connection.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    connection.commit()
    connection.close()


def test_imessage_reader_emits_signals_axes(tmp_path: Path):
    db_path = tmp_path / "chat.db"
    _write_chat_db(db_path)
    manifest = SourceManifest.load(IMESSAGE_MANIFEST)
    manifest.data["default_path"] = str(db_path)

    reader = SourceCLI(ROOT / "sources").reader_for(manifest)
    chunks = list(reader.iter_chunks())

    assert len(chunks) == 1
    chunk = chunks[0]
    assert 'Message from +15555550123 in chat "Chat One"' in chunk.text
    assert chunk.metadata["source_type"] == "imessage"
    assert chunk.metadata["source_id"] == "message:1"
    assert chunk.metadata["source_document_id"] == "chat:1"
    assert chunk.metadata["chunk_seq"] == 0
    assert chunk.metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert chunk.metadata["created_at_synthesized"] is True
    assert chunk.metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert chunk.metadata["modified_at"] == "2026-05-12T11:00:00+00:00"
    assert chunk.metadata["participant_ids"] == ["+15555550123"]
    assert chunk.metadata["mentioned_urls"] == ["https://example.com"]


def test_imessage_reader_validates_required_tables(tmp_path: Path):
    db_path = tmp_path / "chat.db"
    sqlite3.connect(db_path).close()
    manifest = SourceManifest.load(IMESSAGE_MANIFEST)
    manifest.data["default_path"] = str(db_path)

    report = SourceCLI(ROOT / "sources").reader_for(manifest).validate()

    assert report.ok is False
    assert "missing iMessage tables" in report.errors[0]
