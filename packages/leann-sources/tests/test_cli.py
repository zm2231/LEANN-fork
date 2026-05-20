from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources import SourcesPlugin
from leann_sources.registry import write_registry


def _write_source(root: Path, data_root: Path) -> None:
    source_dir = root / "notes" / "local-notes"
    source_dir.mkdir(parents=True)
    data_root.mkdir()
    (data_root / "note.txt").write_text("hello source cli", encoding="utf-8")
    manifest = {
        "name": "local-notes",
        "category": "notes",
        "display_name": "Local Notes",
        "version": "0.1.0",
        "manifest_version": "1.0",
        "data": {
            "type": "filesystem",
            "default_path": str(data_root),
            "glob": "*.txt",
        },
        "auth": {"type": "none"},
        "fields": {
            "created_at": {"source": "stat.st_birthtime", "transform": "unix_to_utc_iso"},
            "modified_at": {"source": "stat.st_mtime", "transform": "unix_to_utc_iso"},
            "source_type": {"value": "document"},
            "source_id": {"source": "relative_path"},
            "source_document_id": {"source": "relative_path"},
            "chunk_seq": {"auto": "row_index_within_source_document_id"},
        },
        "chunking": {"granularity": "file"},
        "privacy": {"tier": "tier_1"},
    }
    (source_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    write_registry(root)


def _parser(plugin: SourcesPlugin) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leann")
    subparsers = parser.add_subparsers(dest="command")
    plugin.register_cli(subparsers, None)
    return parser


def _run(plugin: SourcesPlugin, argv: list[str]) -> None:
    args = _parser(plugin).parse_args(argv)
    assert plugin.handle_cli(args, None) is True


def test_sources_subcommands_against_synthetic_registry(tmp_path: Path, capsys):
    sources_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    _write_source(sources_root, data_root)
    plugin = SourcesPlugin(sources_root=sources_root)

    _run(plugin, ["sources", "list"])
    output = capsys.readouterr().out
    assert "[notes]" in output
    assert "notes/local-notes" in output

    _run(plugin, ["sources", "list", "--category", "notes"])
    assert "local-notes" in capsys.readouterr().out

    _run(plugin, ["sources", "info", "local-notes"])
    assert "data_type: filesystem" in capsys.readouterr().out

    config_dir = tmp_path / "config"
    _run(plugin, ["sources", "install", "local-notes", "--config-dir", str(config_dir)])
    assert (config_dir / "local-notes").is_dir()

    _run(plugin, ["sources", "connect", "local-notes"])
    assert "no auth required" in capsys.readouterr().out

    _run(plugin, ["sources", "validate", "local-notes"])
    assert "local-notes: ok" in capsys.readouterr().out

    _run(plugin, ["sources", "index", "local-notes", "notes-index", "--dry-run"])
    assert "dry-run emitted 1 chunks for notes-index" in capsys.readouterr().out


def test_unified_index_source_routes_to_plugin(tmp_path: Path, capsys):
    sources_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    _write_source(sources_root, data_root)
    plugin = SourcesPlugin(sources_root=sources_root)

    _run(plugin, ["index", "--source", "local-notes", "notes-index", "--dry-run"])

    assert "dry-run emitted 1 chunks for notes-index" in capsys.readouterr().out
