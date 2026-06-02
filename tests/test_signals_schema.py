import asyncio
import importlib.util
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
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
    "chatgpt",
    "claude",
    "claude_code",
    "codex",
    "pi_agent",
    "wechat",
    "whatsapp",
    "github",
    "code",
    "voice_memo",
    "journal",
    "notion",
}
ACTIVITY_TYPES = {"authored", "received", "visited", "modified", "created"}
FULL_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "leann-sources" / "src"))
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


def _core_data_seconds(value: datetime) -> float:
    core_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return value.timestamp() - core_epoch.timestamp()


def test_calendar_chunks_follow_signals_schema(tmp_path):
    from leann_sources.cli import SourceCLI
    from leann_sources.manifest import SourceManifest

    calendar_cache = tmp_path / "Calendar Cache"
    conn = sqlite3.connect(calendar_cache)
    conn.executescript(
        """
        CREATE TABLE CI_EVENT (
            summary TEXT,
            description TEXT,
            location TEXT,
            start_date REAL,
            end_date REAL,
            created_date REAL,
            last_modified_date REAL
        );
        """
    )
    start = datetime(2026, 5, 15, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 15, 15, 0, tzinfo=timezone.utc)
    created = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
    modified = datetime(2026, 5, 12, 13, 0, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO CI_EVENT VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "Temporal planning",
            "Discuss calendar metadata",
            "Conference room",
            _core_data_seconds(start),
            _core_data_seconds(end),
            _core_data_seconds(created),
            _core_data_seconds(modified),
        ),
    )
    conn.commit()
    conn.close()

    sources_root = ROOT / "packages" / "leann-sources" / "sources"
    manifest = SourceManifest.load(sources_root / "calendar" / "apple-calendar" / "manifest.yaml")
    manifest.data["default_path"] = str(calendar_cache)
    chunks = list(SourceCLI(sources_root).reader_for(manifest).iter_chunks())
    documents = [Document(text=chunk.text, metadata=chunk.metadata) for chunk in chunks]

    metadata_rows = asyncio.run(_build_and_read_metadata(tmp_path, "signals-calendar", documents))

    assert metadata_rows
    for metadata in metadata_rows:
        _assert_signals_metadata(metadata)


def _cocoa_ns(value: datetime) -> int:
    cocoa_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return int((value.timestamp() - cocoa_epoch.timestamp()) * 1_000_000_000)


def test_source_registry_imessage_chunks_follow_signals_schema(tmp_path):
    from leann_sources.cli import SourceCLI
    from leann_sources.manifest import SourceManifest

    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
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
    conn.execute("INSERT INTO chat VALUES (1, 'chat-1', 'Chat One')")
    conn.execute("INSERT INTO handle VALUES (1, '+15555550123')")
    conn.execute(
        "INSERT INTO message VALUES (1, 'hello temporal', ?, ?, 0, 'iMessage', 1)",
        (_cocoa_ns(created), _cocoa_ns(edited)),
    )
    conn.execute("INSERT INTO chat_message_join VALUES (1, 1)")
    conn.commit()
    conn.close()

    sources_root = ROOT / "packages" / "leann-sources" / "sources"
    manifest = SourceManifest.load(sources_root / "messaging" / "imessage" / "manifest.yaml")
    manifest.data["default_path"] = str(db_path)
    chunks = list(SourceCLI(sources_root).reader_for(manifest).iter_chunks())
    documents = [Document(text=chunk.text, metadata=chunk.metadata) for chunk in chunks]

    metadata_rows = asyncio.run(_build_and_read_metadata(tmp_path, "signals-imessage", documents))

    assert metadata_rows
    for metadata in metadata_rows:
        _assert_signals_metadata(metadata)


def test_source_registry_catalog_manifests_cover_temporal_axes():
    from leann_sources.cli import SourceCLI
    from leann_sources.registry import build_registry

    sources_root = ROOT / "packages" / "leann-sources" / "sources"
    cli = SourceCLI(sources_root)
    registry = build_registry(sources_root)

    assert len(registry.entries) == 12
    for entry in registry.entries:
        manifest = cli.load_manifest(entry.name)
        assert "created_at" in manifest.fields, entry.name
        assert "modified_at" in manifest.fields, entry.name
        assert "event_time" in manifest.fields, entry.name
