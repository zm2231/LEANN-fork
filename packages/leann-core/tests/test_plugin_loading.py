from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "leann-sources" / "src"))

from leann.cli import LeannCLI
from leann_sources import get_plugin


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
