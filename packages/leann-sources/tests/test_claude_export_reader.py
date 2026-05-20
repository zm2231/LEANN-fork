from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "imports" / "claude-export" / "manifest.yaml"


def test_claude_export_reader_emits_signals_chunks(tmp_path: Path):
    export = tmp_path / "claude.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "Source Registry",
                    "created_at": "2026-05-12T10:00:00Z",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(export)

    chunks = list(SourceCLI(ROOT / "sources").reader_for(manifest).iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "claude"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at_synthesized"] is True
