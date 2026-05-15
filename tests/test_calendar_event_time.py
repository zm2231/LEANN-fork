import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from leann.cli import LeannCLI


class FakeCursor:
    def execute(self, _query, _params):
        return None

    def fetchall(self):
        return [
            (
                123,
                "Temporal planning",
                "Discuss calendar metadata",
                "Conference room",
                "2026-05-15 14:30:00",
                "2026-05-15 10:30:00",
                "2026-05-15 11:00:00",
            )
        ]


class FakeConnection:
    def cursor(self):
        return FakeCursor()

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

    assert event_time.tzinfo is not None
    assert metadata["source_type"] == "calendar"
    assert metadata["source_id"] == "123"
    assert "start" not in metadata
