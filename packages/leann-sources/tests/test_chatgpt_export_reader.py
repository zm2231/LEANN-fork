from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "imports" / "chatgpt-export" / "manifest.yaml"


def test_chatgpt_export_reader_emits_signals_chunks(tmp_path: Path, monkeypatch):
    html = tmp_path / "chat.html"
    html.write_text("<html>placeholder</html>", encoding="utf-8")

    def fake_parse(_self, _html):
        return [
            {
                "title": "Source Registry",
                "timestamp": "2026-05-12T10:00:00Z",
                "messages": [{"role": "user", "content": "hello", "timestamp": None}],
            }
        ]

    monkeypatch.setattr(
        "apps.chatgpt_data.chatgpt_reader.ChatGPTReader._parse_chatgpt_html", fake_parse
    )
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(html)

    chunks = list(SourceCLI(ROOT / "sources").reader_for(manifest).iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "chatgpt"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at_synthesized"] is True
