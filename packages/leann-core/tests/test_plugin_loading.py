from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "leann-sources" / "src"))

from leann.cli import LeannCLI
from leann_sources import SourcesPlugin, get_plugin
from leann_sources.registry import write_registry


@dataclass
class FakeEntryPoint:
    name: str
    target: object

    def load(self):
        return self.target


def test_cli_discovers_leann_sources_plugin(monkeypatch):
    monkeypatch.setattr(
        "leann.cli.importlib.metadata.entry_points",
        lambda group=None: [FakeEntryPoint("sources", get_plugin)]
        if group == "leann.plugins"
        else [],
    )

    cli = LeannCLI()

    assert [plugin.name for plugin in cli.plugins] == ["sources"]


def test_cli_ignores_absent_plugins(monkeypatch):
    monkeypatch.setattr("leann.cli.importlib.metadata.entry_points", lambda group=None: [])

    cli = LeannCLI()
    parser = cli.create_parser()

    assert cli.plugins == []
    assert parser.prog == "leann"


def test_sources_command_only_exists_when_plugin_installed(monkeypatch, tmp_path, capsys):
    sources_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "note.txt").write_text("hello", encoding="utf-8")
    source_dir = sources_root / "notes" / "local-notes"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "local-notes",
                "category": "notes",
                "display_name": "Local Notes",
                "version": "0.1.0",
                "manifest_version": "1.0",
                "data": {"type": "filesystem", "default_path": str(data_root), "glob": "*.txt"},
                "auth": {"type": "none"},
                "fields": {"source_type": {"value": "document"}, "source_id": {"source": "name"}},
                "chunking": {"granularity": "file"},
                "privacy": {"tier": "tier_1"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_registry(sources_root)

    monkeypatch.setattr(
        "leann.cli.importlib.metadata.entry_points",
        lambda group=None: [
            FakeEntryPoint("sources", lambda: SourcesPlugin(sources_root=sources_root))
        ]
        if group == "leann.plugins"
        else [],
    )
    cli = LeannCLI()
    parser = cli.create_parser()
    args = parser.parse_args(["sources", "list"])

    asyncio.run(cli.run(args))

    assert "notes/local-notes" in capsys.readouterr().out

    monkeypatch.setattr("leann.cli.importlib.metadata.entry_points", lambda group=None: [])
    parser_without_plugin = LeannCLI().create_parser()
    with pytest.raises(SystemExit):
        parser_without_plugin.parse_args(["sources", "list"])


def test_index_imessage_alias_routes_to_source_plugin(monkeypatch, tmp_path):
    sources_root = tmp_path / "sources"
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "note.txt").write_text("hello", encoding="utf-8")
    source_dir = sources_root / "messaging" / "imessage"
    source_dir.mkdir(parents=True)
    (source_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "imessage",
                "category": "messaging",
                "display_name": "iMessage",
                "version": "0.1.0",
                "manifest_version": "1.0",
                "data": {"type": "filesystem", "default_path": str(data_root), "glob": "*.txt"},
                "auth": {"type": "none"},
                "fields": {"source_type": {"value": "imessage"}, "source_id": {"source": "name"}},
                "chunking": {"granularity": "file"},
                "privacy": {"tier": "tier_1"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_registry(sources_root)
    monkeypatch.setattr(
        "leann.cli.importlib.metadata.entry_points",
        lambda group=None: [
            FakeEntryPoint("sources", lambda: SourcesPlugin(sources_root=sources_root))
        ]
        if group == "leann.plugins"
        else [],
    )
    cli = LeannCLI()
    parser = cli.create_parser()
    args = parser.parse_args(["index-imessage", "--no-recompute"])
    captured = {}

    async def fake_build(build_args, docs):
        captured["index_name"] = build_args.index_name
        captured["docs"] = docs

    cli._build_index_from_documents = fake_build

    asyncio.run(cli.run(args))

    assert captured["index_name"] == "imessage"
    assert captured["docs"][0].metadata["source_type"] == "imessage"
