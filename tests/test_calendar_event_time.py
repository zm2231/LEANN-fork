import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "leann-sources" / "src"))

from leann_sources.cli import SourceCLI  # noqa: E402
from leann_sources.manifest import SourceManifest  # noqa: E402


def _core_data_seconds(value: datetime) -> float:
    core_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return value.timestamp() - core_epoch.timestamp()


def _load_calendar_metadata(cache: Path):
    sources_root = ROOT / "packages" / "leann-sources" / "sources"
    manifest = SourceManifest.load(sources_root / "calendar" / "apple-calendar" / "manifest.yaml")
    manifest.data["default_path"] = str(cache)
    chunks = list(SourceCLI(sources_root).reader_for(manifest).iter_chunks())
    assert len(chunks) == 1
    return chunks[0].metadata


def test_calendar_reader_emits_event_time_and_source_type(tmp_path):
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

    metadata = _load_calendar_metadata(calendar_cache)
    event_time = datetime.fromisoformat(metadata["event_time"])
    created_at = datetime.fromisoformat(metadata["created_at"])
    modified_at = datetime.fromisoformat(metadata["modified_at"])

    assert event_time.tzinfo is not None
    assert created_at.tzinfo is not None
    assert modified_at.tzinfo is not None
    assert metadata["source_type"] == "calendar"
    assert metadata["source_id"] == "1"
    assert metadata["created_at"] == "2026-05-10T12:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T13:00:00+00:00"
    assert metadata["created_at_synthesized"] is False
    assert metadata["modified_at_synthesized"] is False
    assert "start" not in metadata


def test_calendar_reader_marks_synthesized_axes_when_source_lacks_columns(tmp_path):
    calendar_cache = tmp_path / "Calendar Cache"
    conn = sqlite3.connect(calendar_cache)
    conn.executescript(
        """
        CREATE TABLE CI_EVENT (
            summary TEXT,
            description TEXT,
            location TEXT,
            start_date REAL,
            end_date REAL
        );
        """
    )
    start = datetime(2026, 5, 15, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 15, 15, 0, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO CI_EVENT VALUES (?, ?, ?, ?, ?)",
        (
            "Temporal planning",
            "Discuss calendar metadata",
            "Conference room",
            _core_data_seconds(start),
            _core_data_seconds(end),
        ),
    )
    conn.commit()
    conn.close()

    metadata = _load_calendar_metadata(calendar_cache)
    assert metadata["created_at"] == metadata["event_time"]
    assert metadata["modified_at"] == metadata["event_time"]
    assert metadata["created_at_synthesized"] is True
    assert metadata["modified_at_synthesized"] is True
