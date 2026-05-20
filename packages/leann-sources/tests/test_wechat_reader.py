from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "messaging" / "wechat" / "manifest.yaml"


def test_wechat_reader_emits_signals_chunks(tmp_path: Path):
    export_dir = tmp_path / "wechat"
    export_dir.mkdir()
    (export_dir / "alice.json").write_text(
        json.dumps(
            [
                {
                    "content": "hello source registry",
                    "message": "hello source registry",
                    "createTime": 1778580000,
                    "isSentFromSelf": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["default_path"] = str(export_dir)

    chunks = list(SourceCLI(ROOT / "sources").reader_for(manifest).iter_chunks())

    assert len(chunks) == 1
    metadata = chunks[0].metadata
    assert metadata["source_type"] == "wechat"
    assert metadata["source_id"] == "alice:1778580000"
    assert metadata["created_at"] == "2026-05-12T10:00:00+00:00"
    assert metadata["event_time"] == "2026-05-12T10:00:00+00:00"
    assert metadata["modified_at"] == "2026-05-12T10:00:00+00:00"
