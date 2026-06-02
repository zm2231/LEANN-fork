from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leann_sources.manifest import SourceManifest  # noqa: E402

READER_PATH = ROOT / "sources" / "agent-sessions" / "claude-code" / "reader.py"
MANIFEST_PATH = ROOT / "sources" / "agent-sessions" / "claude-code" / "manifest.yaml"


def _load_reader_module():
    spec = importlib.util.spec_from_file_location("_claude_code_reader_test", READER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def claude_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    project_dir = tmp_path / "-tmp-fixture-project"
    project_dir.mkdir()
    session_path = project_dir / "abcd-1234.jsonl"
    events = [
        {"type": "permission-mode", "permissionMode": "ask"},
        {
            "type": "system",
            "subtype": "turn_duration",
            "cwd": "/tmp/fixture/project",
            "sessionId": "abcd-1234",
            "timestamp": "2026-05-15T14:00:00Z",
        },
        {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": None,
            "timestamp": "2026-05-15T14:30:00Z",
            "message": {"role": "user", "content": "hello claude"},
        },
        {
            "type": "user",
            "uuid": "u-meta",
            "timestamp": "2026-05-15T14:30:01Z",
            "isMeta": True,
            "message": {"role": "user", "content": "<local-command-caveat>skip</local-command-caveat>"},
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "parentUuid": "u-1",
            "timestamp": "2026-05-15T14:30:05Z",
            "message": {
                "model": "claude-sonnet-4-6",
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "considering..."},
                    {"type": "text", "text": "hi alice"},
                    {"type": "tool_use", "name": "Read"},
                ],
            },
        },
    ]
    session_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return tmp_path


def test_claude_code_reader_emits_signals_chunks(claude_root):
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.ClaudeCodeSourceReader(manifest, root=claude_root)
    assert reader.validate().ok

    chunks = list(reader.iter_chunks())
    assert len(chunks) == 2

    user_chunk, assistant_chunk = chunks
    md = user_chunk.metadata
    assert md["source_type"] == "claude_code"
    assert md["author"] == "alice"
    assert md["activity_type"] == "authored"
    assert md["participant_ids"] == ["alice", "claude_code"]
    assert md["source_id"] == "abcd-1234:u-1"
    assert md["source_document_id"] == "agent:claude_code:session:abcd-1234"
    assert md["project_id"] == "project"
    assert md["parent_ref"] == "session:abcd-1234"
    assert md["extra"]["cwd"] == "/tmp/fixture/project"
    assert md["extra"]["agent"] == "claude-code"
    for axis in ("created_at", "modified_at", "event_time"):
        parsed = datetime.fromisoformat(md[axis])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() is not None

    md_a = assistant_chunk.metadata
    assert md_a["author"] == "claude_code"
    assert "[thinking]" in assistant_chunk.text
    assert "[tool_use:Read]" in assistant_chunk.text
    assert md_a["extra"]["model"] == "claude-sonnet-4-6"


def test_claude_code_reader_skips_meta_and_non_message_events(claude_root):
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.ClaudeCodeSourceReader(manifest, root=claude_root)
    chunks = list(reader.iter_chunks())
    sources = {c.metadata["extra"]["uuid"] for c in chunks}
    assert "u-meta" not in sources  # isMeta filtered
    assert "u-1" in sources and "a-1" in sources


def test_claude_code_reader_default_local_user(tmp_path, monkeypatch):
    monkeypatch.delenv("LEANN_LOCAL_USER", raising=False)
    project_dir = tmp_path / "-tmp"
    project_dir.mkdir()
    (project_dir / "s.jsonl").write_text(
        json.dumps({
            "type": "user",
            "uuid": "u",
            "timestamp": "2026-05-15T14:00:00Z",
            "message": {"role": "user", "content": "hi"},
        }) + "\n"
    )
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.ClaudeCodeSourceReader(manifest, root=tmp_path)
    chunks = list(reader.iter_chunks())
    assert chunks[0].metadata["author"] == "local_user"
