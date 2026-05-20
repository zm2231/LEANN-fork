from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "browser" / "chrome" / "manifest.yaml"


def _chrome_time(value: datetime) -> int:
    return int((value.timestamp() + 11644473600) * 1_000_000)


def _write_history(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE urls (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT,
            visit_count INTEGER,
            typed_count INTEGER,
            hidden INTEGER,
            last_visit_time INTEGER
        );
        CREATE TABLE visits (url INTEGER, visit_time INTEGER);
        """
    )
    first = datetime(2026, 5, 10, 10, tzinfo=timezone.utc)
    last = datetime(2026, 5, 12, 10, tzinfo=timezone.utc)
    conn.execute(
        "INSERT INTO urls VALUES (1, 'https://example.com/a', 'Example', 2, 0, 0, ?)",
        (_chrome_time(last),),
    )
    conn.execute("INSERT INTO visits VALUES (1, ?)", (_chrome_time(first),))
    conn.commit()
    conn.close()


def test_chrome_reader_emits_signals_chunks(tmp_path: Path):
    profile = tmp_path / "Default"
    profile.mkdir()
    _write_history(profile / "History")
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(profile)

    chunks = list(SourceCLI(ROOT / "sources").reader_for(manifest).iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "browser_history"
    assert metadata["source_id"] == "https://example.com/a"
    assert metadata["created_at"] == "2026-05-10T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"


def test_chrome_reader_validate_reports_missing_history(tmp_path: Path):
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(tmp_path)

    report = SourceCLI(ROOT / "sources").reader_for(manifest).validate()

    assert report.ok is False
    assert "History" in report.errors[0]
