from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "calendar" / "apple-calendar" / "manifest.yaml"


def _core_data_seconds(value: datetime) -> float:
    core_epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return value.timestamp() - core_epoch.timestamp()


def _write_calendar_cache(path: Path) -> None:
    conn = sqlite3.connect(path)
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
    start = datetime(2026, 5, 12, 10, tzinfo=timezone.utc)
    end = datetime(2026, 5, 12, 11, tzinfo=timezone.utc)
    created = datetime(2026, 5, 1, 9, tzinfo=timezone.utc)
    modified = datetime(2026, 5, 2, 9, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO CI_EVENT VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "Planning",
            "Discuss source registry",
            "Office",
            _core_data_seconds(start),
            _core_data_seconds(end),
            _core_data_seconds(created),
            _core_data_seconds(modified),
        ),
    )
    conn.commit()
    conn.close()


def test_apple_calendar_reader_emits_signals_chunks(tmp_path: Path):
    cache = tmp_path / "Calendar Cache"
    _write_calendar_cache(cache)
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(cache)

    chunks = list(SourceCLI(ROOT / "sources").reader_for(manifest).iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert "Event: Planning" in chunks[0].text
    assert metadata["source_type"] == "calendar"
    assert metadata["source_id"] == "1"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["created_at"] == "2026-05-01T09:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-02T09:00:00+00:00"


def test_apple_calendar_reader_validate_reports_missing_cache(tmp_path: Path):
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(tmp_path / "missing")

    report = SourceCLI(ROOT / "sources").reader_for(manifest).validate()

    assert report.ok is False
    assert "does not exist" in report.errors[0]
