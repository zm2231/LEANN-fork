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

READER_PATH = ROOT / "sources" / "agent-sessions" / "pi-agent" / "reader.py"
MANIFEST_PATH = ROOT / "sources" / "agent-sessions" / "pi-agent" / "manifest.yaml"


def _load_reader_module():
    spec = importlib.util.spec_from_file_location("_pi_agent_reader_test", READER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_session(root: Path, cwd: str, session_id: str, *, custom_display=True) -> Path:
    subdir = root / cwd.replace("/", "-")
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"2026-05-15T14-00-00_{session_id}.jsonl"
    events = [
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": "2026-05-15T14:00:00Z",
            "cwd": cwd,
        },
        {
            "type": "model_change",
            "id": "mc-1",
            "timestamp": "2026-05-15T14:00:00Z",
            "provider": "openai-codex",
            "modelId": "gpt-5.2-codex",
        },
        {
            "type": "thinking_level_change",
            "id": "tl-1",
            "timestamp": "2026-05-15T14:00:00Z",
            "thinkingLevel": "medium",
        },
        {
            "type": "message",
            "id": "m-1",
            "parentId": None,
            "timestamp": "2026-05-15T14:00:10Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "hi pi"}]},
        },
        {
            "type": "custom_message",
            "id": "c-1",
            "customType": "zoe-steer",
            "timestamp": "2026-05-15T14:00:11Z",
            "content": "policy here",
            "display": custom_display,
        },
        {
            "type": "message",
            "id": "m-2",
            "parentId": "m-1",
            "timestamp": "2026-05-15T14:00:12Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


@pytest.fixture
def pi_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    monkeypatch.delenv("LEANN_PI_AGENT_ROOTS", raising=False)
    _write_session(tmp_path, "/tmp/fixture/my-repo", "sess-1")
    return tmp_path


def test_pi_agent_reader_emits_signals_chunks(pi_root):
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.PiAgentSourceReader(manifest, roots=[pi_root])
    assert reader.validate().ok

    chunks = list(reader.iter_chunks())
    # user message + custom_message (display=True) + assistant message = 3
    assert len(chunks) == 3

    user_chunk = next(c for c in chunks if c.metadata["author"] == "alice")
    assert user_chunk.metadata["source_type"] == "pi_agent"
    assert user_chunk.metadata["source_document_id"] == "agent:pi_agent:session:sess-1"
    assert user_chunk.metadata["project_id"] == "my-repo"
    assert user_chunk.metadata["extra"]["model_id"] == "gpt-5.2-codex"
    assert user_chunk.metadata["extra"]["thinking_level"] == "medium"
    assert user_chunk.metadata["extra"]["provider"] == "openai-codex"
    for axis in ("created_at", "modified_at", "event_time"):
        parsed = datetime.fromisoformat(user_chunk.metadata[axis])
        assert parsed.tzinfo is not None and parsed.utcoffset() is not None

    custom_chunk = next(c for c in chunks if c.metadata["extra"]["event_type"] == "custom_message")
    assert custom_chunk.metadata["extra"]["custom_type"] == "zoe-steer"
    assert "[custom_message:zoe-steer]" in custom_chunk.text

    source_ids = [c.metadata["source_id"] for c in chunks]
    assert len(set(source_ids)) == len(source_ids)
    assert any(s.endswith(":message:m-1") for s in source_ids)
    assert any(s.endswith(":custom_message:c-1") for s in source_ids)


def test_pi_agent_reader_skips_display_false(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    _write_session(tmp_path, "/tmp/fixture/my-repo", "sess-2", custom_display=False)
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    reader = mod.PiAgentSourceReader(manifest, roots=[tmp_path])
    chunks = list(reader.iter_chunks())
    event_types = {c.metadata["extra"]["event_type"] for c in chunks}
    assert "custom_message" not in event_types


def test_pi_agent_reader_dedupes_overlapping_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    _write_session(tmp_path, "/tmp/fixture/my-repo", "sess-3")
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    nested = tmp_path / "--tmp-fixture-my-repo--"
    reader = mod.PiAgentSourceReader(manifest, roots=[tmp_path, nested])
    files = list(reader._iter_session_files())
    assert len(files) == 1


def test_pi_agent_reader_extra_roots_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANN_LOCAL_USER", "alice")
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    primary.mkdir()
    secondary.mkdir()
    _write_session(primary, "/tmp/fixture/repo-a", "sess-a")
    _write_session(secondary, "/tmp/fixture/repo-b", "sess-b")
    monkeypatch.setenv("LEANN_PI_AGENT_ROOTS", str(secondary))
    mod = _load_reader_module()
    manifest = SourceManifest.load(MANIFEST_PATH)
    manifest.data["roots"] = [str(primary)]
    reader = mod.PiAgentSourceReader(manifest)
    chunks = list(reader.iter_chunks())
    projects = {c.metadata["project_id"] for c in chunks}
    assert projects == {"repo-a", "repo-b"}
