from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "email" / "apple-mail" / "manifest.yaml"


def _write_emlx(path: Path) -> None:
    path.write_text(
        "1\n"
        "From: alice@example.com\n"
        "To: bob@example.com\n"
        "Subject: Source registry\n"
        "Date: Tue, 12 May 2026 10:00:00 +0000\n"
        "X-Last-Modified: Tue, 12 May 2026 11:00:00 +0000\n"
        "\n"
        "Body with https://example.com\n",
        encoding="utf-8",
    )


def test_apple_mail_reader_emits_signals_chunks(tmp_path: Path):
    messages = tmp_path / "V10" / "Account" / "Mailbox.mbox" / "Messages"
    messages.mkdir(parents=True)
    _write_emlx(messages / "1.emlx")
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(tmp_path)

    reader = SourceCLI(ROOT / "sources").reader_for(manifest)
    chunks = list(reader.iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "email"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T11:00:00+00:00"
    assert metadata["author"] == "alice@example.com"
    assert metadata["activity_type"] == "authored"


def test_apple_mail_reader_validate_reports_missing_messages(tmp_path: Path):
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(tmp_path)

    report = SourceCLI(ROOT / "sources").reader_for(manifest).validate()

    assert report.ok is False
    assert "no Apple Mail Messages directories" in report.errors[0]
