import argparse
import asyncio
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from leann.cli import LeannCLI
from llama_index.core import Document

SOURCE_TYPES = {
    "document",
    "git_commit",
    "slack",
    "daily_summary",
    "email",
    "imessage",
    "browser_history",
    "calendar",
    "code",
    "voice_memo",
    "journal",
    "notion",
}
ACTIVITY_TYPES = {"authored", "received", "visited", "modified", "created"}
FULL_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
ROOT = Path(__file__).resolve().parents[1]
BUILD_EVAL_CORPUS_PATH = ROOT / "scripts" / "build_eval_corpus.py"
spec = importlib.util.spec_from_file_location("build_eval_corpus", BUILD_EVAL_CORPUS_PATH)
assert spec is not None
build_eval_corpus = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(build_eval_corpus)


def _parse_aware_iso(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None
    return parsed


def _assert_no_naive_datetime(value):
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_naive_datetime(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_naive_datetime(nested)
    elif isinstance(value, str) and FULL_DATETIME_RE.match(value):
        _parse_aware_iso(value)


def _assert_signals_metadata(metadata):
    assert metadata["source_type"] in SOURCE_TYPES

    if "event_time" in metadata:
        event_time = _parse_aware_iso(metadata["event_time"])
        assert event_time.utcoffset().total_seconds() == 0

    indexed_at = _parse_aware_iso(metadata["indexed_at"])
    assert indexed_at.utcoffset().total_seconds() == 0

    author = metadata.get("author")
    if author:
        assert metadata["activity_type"] in ACTIVITY_TYPES

    participant_ids = metadata.get("participant_ids")
    if participant_ids is not None:
        assert isinstance(participant_ids, list)
        if author:
            assert author in participant_ids

    _assert_no_naive_datetime(metadata)


async def _build_and_read_metadata(tmp_path, index_name, documents):
    cli = LeannCLI()
    cli.indexes_dir = tmp_path / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True, exist_ok=True)
    cli.register_project_dir = lambda: None
    args = SimpleNamespace(
        index_name=index_name,
        embedding_model="all-MiniLM-L6-v2",
        embedding_mode="sentence-transformers",
        no_recompute=True,
    )

    await cli._build_index_from_documents(args, documents)

    passages_path = Path(cli.get_index_path(index_name) + ".passages.jsonl")
    with passages_path.open(encoding="utf-8") as f:
        return [json.loads(line)["metadata"] for line in f if line.strip()]


def test_document_chunks_follow_signals_schema(tmp_path):
    documents = [
        Document(
            text="A short document chunk with a timestamp.",
            metadata={
                "source_type": "document",
                "source_id": "doc-1",
                "event_time": "2026-05-15T14:30:00+00:00",
                "mentioned_urls": [],
                "mentioned_refs": [],
                "mentioned_files": [],
            },
        )
    ]

    metadata_rows = asyncio.run(_build_and_read_metadata(tmp_path, "signals-docs", documents))

    assert metadata_rows
    for metadata in metadata_rows:
        _assert_signals_metadata(metadata)


def test_git_commit_chunks_from_eval_builder_follow_signals_schema(tmp_path, monkeypatch):
    def fake_check_output(command, **kwargs):
        if command[:2] == ["git", "log"]:
            return (
                "abc123\x1f2026-05-01T12:00:00+00:00\x1f"
                "2026-05-01T13:00:00+00:00\x1fZain\x1fAdd temporal filters #12\x1f"
                "Body mentions https://example.com\x1e"
            )
        if command[:2] == ["git", "show"]:
            return "packages/leann-core/src/leann/api.py\n"
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(build_eval_corpus.subprocess, "check_output", fake_check_output)

    documents = build_eval_corpus.ingest_commits()
    metadata_rows = asyncio.run(_build_and_read_metadata(tmp_path, "signals-git", documents))

    assert metadata_rows
    for metadata in metadata_rows:
        _assert_signals_metadata(metadata)


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


def test_calendar_chunks_follow_signals_schema(tmp_path, monkeypatch):
    home = tmp_path / "home"
    calendar_cache = home / "Library" / "Calendars" / "Calendar Cache"
    calendar_cache.parent.mkdir(parents=True)
    calendar_cache.write_text("sqlite placeholder", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr("shutil.copy2", lambda _src, _dst: None)
    monkeypatch.setattr("sqlite3.connect", lambda _path: FakeConnection())

    captured = {}

    async def capture_build(_args, docs):
        captured["docs"] = docs

    cli = LeannCLI()
    cli._build_index_from_documents = capture_build

    asyncio.run(cli.index_calendar(argparse.Namespace(max_count=1)))

    metadata_rows = asyncio.run(
        _build_and_read_metadata(tmp_path, "signals-calendar", captured["docs"])
    )

    assert metadata_rows
    for metadata in metadata_rows:
        _assert_signals_metadata(metadata)
