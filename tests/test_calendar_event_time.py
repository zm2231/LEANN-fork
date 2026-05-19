import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from leann.cli import LeannCLI


class FakeCursor:
    def __init__(self):
        self.query = ""

    def execute(self, query, _params=None):
        self.query = query
        return None

    def fetchall(self):
        if "PRAGMA table_info" in self.query:
            return [
                (0, "created_date", "REAL", 0, None, 0),
                (1, "last_modified_date", "REAL", 0, None, 0),
            ]
        return [
            (
                123,
                "Temporal planning",
                "Discuss calendar metadata",
                "Conference room",
                "2026-05-15 14:30:00",
                "2026-05-15 10:30:00",
                "2026-05-15 11:00:00",
                "2026-05-10 12:00:00",
                "2026-05-12 13:00:00",
            )
        ]


class FakeConnection:
    def cursor(self):
        return FakeCursor()

    def close(self):
        return None


class SparseFakeCursor(FakeCursor):
    def fetchall(self):
        if "PRAGMA table_info" in self.query:
            return []
        return [
            (
                123,
                "Temporal planning",
                "Discuss calendar metadata",
                "Conference room",
                "2026-05-15 14:30:00",
                "2026-05-15 10:30:00",
                "2026-05-15 11:00:00",
                None,
                None,
            )
        ]


class SparseFakeConnection:
    def cursor(self):
        return SparseFakeCursor()

    def close(self):
        return None


def test_calendar_reader_emits_event_time_and_source_type(tmp_path, monkeypatch):
    home = tmp_path / "home"
    calendar_cache = home / "Library" / "Calendars" / "Calendar Cache"
    calendar_cache.parent.mkdir(parents=True)
    calendar_cache.write_text("sqlite placeholder", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr("shutil.copy2", lambda _src, _dst: None)
    monkeypatch.setattr("sqlite3.connect", lambda _path: FakeConnection())

    captured = {}

    async def fake_build(_args, docs):
        captured["docs"] = docs

    cli = LeannCLI()
    cli._build_index_from_documents = fake_build

    asyncio.run(cli.index_calendar(argparse.Namespace(max_count=1)))

    docs = captured["docs"]
    assert len(docs) == 1
    metadata = docs[0].metadata
    event_time = datetime.fromisoformat(metadata["event_time"])
    created_at = datetime.fromisoformat(metadata["created_at"])
    modified_at = datetime.fromisoformat(metadata["modified_at"])

    assert event_time.tzinfo is not None
    assert created_at.tzinfo is not None
    assert modified_at.tzinfo is not None
    assert metadata["source_type"] == "calendar"
    assert metadata["source_id"] == "123"
    assert metadata["created_at"] == "2026-05-10T12:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T13:00:00+00:00"
    assert metadata["created_at_synthesized"] is False
    assert metadata["modified_at_synthesized"] is False
    assert "start" not in metadata


def test_calendar_reader_marks_synthesized_axes_when_source_lacks_columns(tmp_path, monkeypatch):
    home = tmp_path / "home"
    calendar_cache = home / "Library" / "Calendars" / "Calendar Cache"
    calendar_cache.parent.mkdir(parents=True)
    calendar_cache.write_text("sqlite placeholder", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr("shutil.copy2", lambda _src, _dst: None)
    monkeypatch.setattr("sqlite3.connect", lambda _path: SparseFakeConnection())

    captured = {}

    async def fake_build(_args, docs):
        captured["docs"] = docs

    cli = LeannCLI()
    cli._build_index_from_documents = fake_build

    asyncio.run(cli.index_calendar(argparse.Namespace(max_count=1)))

    metadata = captured["docs"][0].metadata
    assert metadata["created_at"] == metadata["event_time"]
    assert metadata["modified_at"] == metadata["event_time"]
    assert metadata["created_at_synthesized"] is True
    assert metadata["modified_at_synthesized"] is True
