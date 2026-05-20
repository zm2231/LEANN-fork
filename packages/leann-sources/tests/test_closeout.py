from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources.cli import SourceCLI
from leann_sources.registry import build_registry

ROOT = Path(__file__).resolve().parents[1]
SOURCES_ROOT = ROOT / "sources"
EXPECTED_SOURCES = {
    "apple-calendar",
    "apple-mail",
    "chatgpt-export",
    "chrome",
    "claude-export",
    "github",
    "imessage",
    "wechat",
    "whatsapp",
}


def test_closeout_registry_has_nine_sources_and_grouped_list(capsys):
    registry = build_registry(SOURCES_ROOT)

    assert {entry.name for entry in registry.entries} == EXPECTED_SOURCES
    assert len(registry.entries) == 9

    SourceCLI(SOURCES_ROOT).list_sources()
    output = capsys.readouterr().out

    assert "[browser]" in output
    assert "[developer-tools]" in output
    assert "developer-tools/github" in output
    assert "messaging/whatsapp" in output


def test_closeout_manifests_declare_temporal_axes():
    registry = build_registry(SOURCES_ROOT)
    cli = SourceCLI(SOURCES_ROOT)

    for entry in registry.entries:
        manifest = cli.load_manifest(entry.name)
        assert "created_at" in manifest.fields, entry.name
        assert "modified_at" in manifest.fields, entry.name
        assert "event_time" in manifest.fields, entry.name
        assert manifest.fields["source_type"]["value"]


def test_closeout_validate_reports_human_readable_status(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORIES", raising=False)
    cli = SourceCLI(SOURCES_ROOT)
    registry = build_registry(SOURCES_ROOT)

    for entry in registry.entries:
        cli.validate(entry.name)

    output = capsys.readouterr().out
    for name in EXPECTED_SOURCES:
        assert f"{name}:" in output
    assert "github: missing auth" in output
