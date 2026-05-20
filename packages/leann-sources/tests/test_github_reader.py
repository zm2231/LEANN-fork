from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.manifest import SourceManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "developer-tools" / "github" / "manifest.yaml"


def test_github_reader_validates_fixture_and_emits_signals_chunks(monkeypatch):
    manifest = SourceManifest.load(MANIFEST)
    manifest.data["pages"] = [
        [
            {
                "kind": "issue",
                "repository": "openai/leann",
                "number": 287,
                "node_id": "I_287",
                "title": "Source registry",
                "body": "Track #12 and docs/SOURCES.md at https://example.com/spec",
                "html_url": "https://github.com/openai/leann/issues/287",
                "created_at": "2026-05-02T10:00:00Z",
                "updated_at": "2026-05-02T11:00:00Z",
                "user": {"login": "alice"},
                "mentioned_files": ["docs/SOURCES.md"],
            },
            {
                "kind": "commit",
                "repository": "openai/leann",
                "sha": "abc123",
                "html_url": "https://github.com/openai/leann/commit/abc123",
                "commit": {
                    "message": "Add source registry\n\nTouches packages/leann-sources.",
                    "author": {
                        "name": "Bob",
                        "email": "bob@example.com",
                        "date": "2026-05-03T12:00:00Z",
                    },
                    "committer": {
                        "name": "Bob",
                        "email": "bob@example.com",
                        "date": "2026-05-03T12:05:00Z",
                    },
                },
                "files": [{"filename": "packages/leann-sources/src/leann_sources/registry.py"}],
            },
        ]
    ]
    monkeypatch.setenv("GITHUB_TOKEN", "fixture-token")

    reader = SourceCLI(ROOT / "sources").reader_for(manifest)
    assert reader.validate().ok

    chunks = list(reader.iter_chunks())

    assert len(chunks) == 2
    issue = chunks[0].metadata
    assert issue["source_type"] == "github"
    assert issue["source_id"] == "github:openai/leann:issue:I_287"
    assert issue["project_id"] == "openai/leann"
    assert issue["parent_ref"] == "github:openai/leann#287"
    assert issue["event_time"] == "2026-05-02T10:00:00Z"
    assert issue["event_time_synthesized"] is True
    assert issue["mentioned_urls"] == ["https://example.com/spec"]
    assert issue["mentioned_refs"] == ["#12"]
    assert issue["mentioned_files"] == ["docs/SOURCES.md"]

    commit = chunks[1].metadata
    assert commit["source_id"] == "github:openai/leann:commit:abc123"
    assert commit["source_document_id"] == "github:openai/leann:commits"
    assert commit["mentioned_files"] == ["packages/leann-sources/src/leann_sources/registry.py"]
