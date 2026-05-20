from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "messaging" / "whatsapp" / "manifest.yaml"


def _make_chatstorage(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE ZWACHATSESSION (
              Z_PK INTEGER PRIMARY KEY,
              ZCONTACTJID TEXT,
              ZPARTNERNAME TEXT,
              ZGROUPNAME TEXT
            );
            CREATE TABLE ZWAMESSAGE (
              Z_PK INTEGER PRIMARY KEY,
              ZCHATSESSION INTEGER,
              ZTEXT TEXT,
              ZMESSAGEDATE REAL,
              ZEDITEDDATE REAL,
              ZISFROMME INTEGER,
              ZFROMJID TEXT,
              ZCONTACTJID TEXT
            );
            """
        )
        connection.execute(
            "INSERT INTO ZWACHATSESSION VALUES (?, ?, ?, ?)",
            (7, "alice@example.whatsapp", "Alice", None),
        )
        connection.execute(
            "INSERT INTO ZWAMESSAGE VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                42,
                7,
                "source registry link https://example.com",
                799_236_000,
                799_236_060,
                0,
                "alice@example.whatsapp",
                "alice@example.whatsapp",
            ),
        )


def test_whatsapp_reader_validates_and_emits_signals_chunks(tmp_path: Path):
    db_path = tmp_path / "ChatStorage.sqlite"
    _make_chatstorage(db_path)
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(db_path)

    reader = SourceCLI(ROOT / "sources").reader_for(manifest)
    assert reader.validate().ok

    chunks = list(reader.iter_chunks())

    assert len(chunks) == 1
    assert "WhatsApp message from alice@example.whatsapp" in chunks[0].text
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "whatsapp"
    assert metadata["source_id"] == "whatsapp_message:42"
    assert metadata["source_document_id"] == "whatsapp_chat:7"
    assert metadata["event_time"] == "2026-04-30T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-04-30T10:01:00+00:00"
    assert metadata["created_at_synthesized"] is True
    assert metadata["participant_ids"] == ["alice@example.whatsapp"]
    assert metadata["mentioned_urls"] == ["https://example.com"]
