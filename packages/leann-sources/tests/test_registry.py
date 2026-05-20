from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leann_sources import get_plugin
from leann_sources.registry import build_registry, discover_manifests, write_registry


def _write_manifest(root: Path, category: str, name: str, data_type: str = "filesystem") -> None:
    source_dir = root / category / name
    source_dir.mkdir(parents=True)
    manifest = {
        "name": name,
        "category": category,
        "display_name": name.replace("-", " ").title(),
        "version": "0.1.0",
        "manifest_version": "1.0",
        "data": {"type": data_type, "default_path": f"~/.fixture/{name}"},
        "auth": {"type": "none"},
        "fields": {
            "source_type": {"value": name},
            "source_id": {"source": "id"},
        },
        "chunking": {"granularity": "message"},
        "privacy": {"tier": "tier_1"},
    }
    (source_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def test_registry_discovers_and_groups_sources(tmp_path: Path):
    _write_manifest(tmp_path, "messaging", "imessage", "sqlite")
    _write_manifest(tmp_path, "email", "apple-mail")
    _write_manifest(tmp_path, "developer-tools", "github", "api")

    manifests = discover_manifests(tmp_path)
    registry = build_registry(tmp_path)

    assert [manifest.name for manifest in manifests] == ["github", "apple-mail", "imessage"]
    assert [entry.name for entry in registry.entries] == ["github", "apple-mail", "imessage"]
    assert set(registry.by_category()) == {"developer-tools", "email", "messaging"}
    assert registry.entries[0].manifest_path == "developer-tools/github/manifest.yaml"
    assert registry.entries[0].data_type == "api"


def test_write_registry_outputs_yaml(tmp_path: Path):
    _write_manifest(tmp_path, "messaging", "imessage", "sqlite")
    output_path = tmp_path / "registry.yaml"

    registry = write_registry(tmp_path, output_path)
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert len(registry.entries) == 1
    assert data["manifest_version"] == "1.0"
    assert data["sources"][0]["name"] == "imessage"


def test_get_plugin_exposes_registry_descriptor():
    plugin = get_plugin()

    assert plugin.name == "sources"
    assert plugin.registry_path.name == "registry.yaml"
    assert plugin.index_source_handler is None
