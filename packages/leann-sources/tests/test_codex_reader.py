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

READER_PATH = ROOT / "sources" / "agent-sessions" / "codex" / "reader.py"
MANIFEST_PATH = ROOT / "sources" / "agent-sessions" / "codex" / "manifest.yaml"


def _load_reader_module():
    spec = importlib.util.spec_from_file_location("_codex_reader_test", READER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def codex_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    sessions = tmp_path / "sessions" / "2026" / "05" / "15"
    sessions.mkdir(parents=True)
    rollout = sessions / "rollout-2026-05-15T14-00-00-019abc.jsonl"
    events = [
        {
            "timestamp": "2026-05-15T14:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": "019abc-7000-0001",
                "timestamp": "2026-05-15T14:00:00Z",
                "cwd": "/tmp/fixture/Study-Sync",
                "cli_version": "0.80.0",
                "originator": "codex_cli_rs",
            },
        },
        {
            "timestamp": "2026-05-15T14:00:01Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "find the bug"}],
            },
        },
        {
            "timestamp": "2026-05-15T14:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-A",
                "name": "shell",
                "arguments": "{\"cmd\":\"grep -n foo .\"}",
            },
        },
        {
            "timestamp": "2026-05-15T14:00:03Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-A",
                "output": "x" * 5000,  # exceeds truncation cap
            },
        },
        {
            "timestamp": "2026-05-15T14:00:04Z",
            "type": "event_msg",
            "payload": {"type": "system"},
        },
    ]
    rollout.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return tmp_path / "sessions"


def test_codex_reader_emits_signals_chunks(codex_root):
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.CodexSourceReader(manifest, root=codex_root)
    assert reader.validate().ok

    chunks = list(reader.iter_chunks())
    # message + function_call + function_call_output = 3 (event_msg skipped)
    assert len(chunks) == 3

    source_ids = [c.metadata["source_id"] for c in chunks]
    assert len(set(source_ids)) == 3  # call/call_output must not collide
    assert any(s.endswith(":message:msg-1") for s in source_ids)
    assert any(s.endswith(":function_call:call-A") for s in source_ids)
    assert any(s.endswith(":function_call_output:call-A") for s in source_ids)

    for chunk in chunks:
        md = chunk.metadata
        assert md["source_type"] == "codex"
        assert md["source_document_id"] == "agent:codex:session:019abc-7000-0001"
        assert md["project_id"] == "Study-Sync"
        for axis in ("created_at", "modified_at", "event_time"):
            parsed = datetime.fromisoformat(md[axis])
            assert parsed.tzinfo is not None and parsed.utcoffset() is not None

    output_chunk = next(c for c in chunks if ":function_call_output:" in c.metadata["source_id"])
    assert output_chunk.metadata["extra"]["truncated"] is True
    assert output_chunk.metadata["extra"]["original_length"] > 4000
    assert len(output_chunk.text) <= 4100  # 4000 + truncation marker length

    message_chunk = next(c for c in chunks if ":message:msg-1" in c.metadata["source_id"])
    assert message_chunk.metadata["author"] == "alice"
    assert message_chunk.metadata["activity_type"] == "authored"


def test_codex_reader_default_local_user(codex_root, monkeypatch):
    monkeypatch.delenv("LEANN_LOCAL_USER", raising=False)
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.CodexSourceReader(manifest, root=codex_root)
    chunks = list(reader.iter_chunks())
    message_chunk = next(c for c in chunks if ":message:msg-1" in c.metadata["source_id"])
    assert message_chunk.metadata["author"] == "local_user"


def test_codex_reader_skips_event_msg_and_turn_context(codex_root):
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.CodexSourceReader(manifest, root=codex_root)
    chunks = list(reader.iter_chunks())
    payload_types = {c.metadata["extra"]["payload_type"] for c in chunks}
    assert "system" not in payload_types
    assert payload_types == {"message", "function_call", "function_call_output"}
