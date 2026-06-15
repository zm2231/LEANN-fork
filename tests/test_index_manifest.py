"""Tests for the global index manifest that powers fast ``leann list``.

The manifest (``~/.leann/indexes.json``) is upserted on index build and
forgotten on remove, so ``leann list`` reads a cache instead of ``os.walk``-ing
every registered project tree. These tests cover the manifest module's record/
forget/prune logic, the CLI fast-path, bounded first-run seeding, and
non-destructive ``--refresh``.
"""

from __future__ import annotations

import importlib
import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import pytest


@pytest.fixture()
def manifest(tmp_path, monkeypatch):
    """Isolate ~/.leann into a temp HOME and return a fresh manifest module."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import leann.index_manifest as m

    importlib.reload(m)
    return m


def _make_cli_index(root: Path, project: str, name: str, mb: int) -> Path:
    d = root / project / ".leann" / "indexes" / name
    d.mkdir(parents=True)
    meta = d / "documents.leann.meta.json"
    meta.write_text("{}")
    (d / "documents.leann").write_bytes(b"x" * (mb * 1_000_000))
    return meta


def _make_app_index(root: Path, project: str, file_base: str, mb: int) -> Path:
    p = root / project
    p.mkdir(parents=True, exist_ok=True)
    (p / ".leann").mkdir(exist_ok=True)
    meta = p / f"{file_base}.leann.meta.json"
    meta.write_text("{}")
    (p / f"{file_base}.leann").write_bytes(b"y" * (mb * 1_000_000))
    return meta


def test_record_cli_and_app(tmp_path, manifest):
    meta_cli = _make_cli_index(tmp_path, "proj-a", "coding-sessions", 2)
    meta_app = _make_app_index(tmp_path, "proj-b", "myidx", 1)

    manifest.record_index(meta_cli, backend="hnsw", embedding_model="BAAI/bge-m3")
    manifest.record_index(meta_app, backend="hnsw")

    entries = {e["type"]: e for e in manifest.load_manifest()["indexes"].values()}
    assert entries["cli"]["name"] == "coding-sessions"
    assert entries["cli"]["project"] == str((tmp_path / "proj-a").resolve())
    assert entries["cli"]["size_mb"] > 1.5
    assert entries["cli"]["backend"] == "hnsw"
    # app display name is the parent directory name
    assert entries["app"]["name"] == "proj-b"
    assert entries["app"]["project"] == str((tmp_path / "proj-b").resolve())


def test_iter_prunes_deleted(tmp_path, manifest):
    meta = _make_cli_index(tmp_path, "proj-a", "idx", 1)
    manifest.record_index(meta)
    assert len(list(manifest.iter_indexes())) == 1

    shutil.rmtree(tmp_path / "proj-a")
    assert list(manifest.iter_indexes()) == []  # stale dropped
    assert manifest.load_manifest()["indexes"] == {}  # and persisted


def test_forget_index_and_dir(tmp_path, manifest):
    meta_app = _make_app_index(tmp_path, "proj-b", "myidx", 1)
    meta_cli = _make_cli_index(tmp_path, "proj-a", "idx", 1)
    manifest.record_index(meta_app)
    manifest.record_index(meta_cli)

    manifest.forget_index(meta_app)
    assert len(manifest.load_manifest()["indexes"]) == 1

    manifest.forget_dir(meta_cli.parent)
    assert manifest.load_manifest()["indexes"] == {}


def test_list_fast_path_render(tmp_path, manifest, monkeypatch):
    from leann.cli import LeannCLI

    meta_a = _make_cli_index(tmp_path, "proj-a", "coding-sessions", 3)
    meta_b = _make_cli_index(tmp_path, "proj-b", "anthropic-workshops", 1)
    manifest.record_index(meta_a)
    manifest.record_index(meta_b)

    monkeypatch.chdir(tmp_path / "proj-a")
    buf = io.StringIO()
    with redirect_stdout(buf):
        LeannCLI().list_indexes()
    out = buf.getvalue()

    assert "🏠 Current Project" in out
    assert "1. 📁 coding-sessions ✅" in out  # current project's index
    assert "anthropic-workshops" in out  # other project's index
    assert "Total: 2 indexes across 2 projects" in out
    assert 'leann search coding-sessions "your query"' in out


def test_empty_manifest_list_seeds_cli_without_app_scan(tmp_path, manifest, monkeypatch):
    from leann.cli import LeannCLI

    _make_cli_index(tmp_path, "proj-a", "fast-cli", 1)
    _make_app_index(tmp_path, "proj-a", "legacy-app", 1)

    (tmp_path / ".leann").mkdir(exist_ok=True)
    (tmp_path / ".leann" / "projects.json").write_text(
        json.dumps([str(tmp_path / "proj-a")])
    )

    monkeypatch.chdir(tmp_path / "proj-a")
    buf = io.StringIO()
    with redirect_stdout(buf):
        LeannCLI().list_indexes()
    out = buf.getvalue()

    assert "fast-cli" in out
    assert "legacy-app" not in out

    names = sorted(e["name"] for e in manifest.load_manifest()["indexes"].values())
    assert names == ["fast-cli"]

    with redirect_stdout(io.StringIO()):
        LeannCLI().list_indexes(refresh=True)

    names = sorted(e["name"] for e in manifest.load_manifest()["indexes"].values())
    assert names == ["fast-cli", "proj-a"]


def test_refresh_is_non_destructive(tmp_path, manifest, monkeypatch):
    """--refresh scans only registered projects but must not drop a still-on-disk
    index recorded for an unregistered project."""
    from leann.cli import LeannCLI

    meta_a = _make_cli_index(tmp_path, "proj-a", "registered", 1)
    meta_b = _make_cli_index(tmp_path, "proj-b", "unregistered", 1)
    manifest.record_index(meta_a)
    manifest.record_index(meta_b)

    # Register only proj-a
    (tmp_path / ".leann").mkdir(exist_ok=True)
    (tmp_path / ".leann" / "projects.json").write_text(json.dumps([str(tmp_path / "proj-a")]))

    monkeypatch.chdir(tmp_path)
    with redirect_stdout(io.StringIO()):
        LeannCLI().list_indexes(refresh=True)

    names = sorted(e["name"] for e in manifest.load_manifest()["indexes"].values())
    assert names == ["registered", "unregistered"]  # proj-b survived
