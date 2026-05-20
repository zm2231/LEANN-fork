"""Manifest discovery and registry generation for LEANN sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from leann_sources.manifest import SourceManifest

DEFAULT_SOURCES_ROOT = Path(__file__).resolve().parents[2] / "sources"


@dataclass(frozen=True)
class SourceRegistryEntry:
    name: str
    category: str
    display_name: str
    version: str
    manifest_path: str
    data_type: str
    privacy_tier: str

    @classmethod
    def from_manifest(cls, manifest: SourceManifest, *, root: Path) -> SourceRegistryEntry:
        if manifest.path is None:
            raise ValueError("manifest path is required for registry entries")
        return cls(
            name=manifest.name,
            category=manifest.category,
            display_name=manifest.display_name,
            version=manifest.version,
            manifest_path=manifest.path.relative_to(root).as_posix(),
            data_type=str(manifest.data["type"]),
            privacy_tier=str(manifest.privacy["tier"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "display_name": self.display_name,
            "version": self.version,
            "manifest_path": self.manifest_path,
            "data_type": self.data_type,
            "privacy_tier": self.privacy_tier,
        }


@dataclass(frozen=True)
class SourceRegistry:
    root: Path
    entries: tuple[SourceRegistryEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": "1.0",
            "sources": [entry.to_dict() for entry in self.entries],
        }

    def by_category(self) -> dict[str, list[SourceRegistryEntry]]:
        categories: dict[str, list[SourceRegistryEntry]] = {}
        for entry in self.entries:
            categories.setdefault(entry.category, []).append(entry)
        return categories


def discover_manifests(root: str | Path = DEFAULT_SOURCES_ROOT) -> list[SourceManifest]:
    sources_root = Path(root)
    manifests = []
    for path in sorted(sources_root.glob("**/manifest.yaml")):
        manifests.append(SourceManifest.load(path))
    return manifests


def build_registry(root: str | Path = DEFAULT_SOURCES_ROOT) -> SourceRegistry:
    sources_root = Path(root)
    entries = [
        SourceRegistryEntry.from_manifest(manifest, root=sources_root)
        for manifest in discover_manifests(sources_root)
    ]
    entries.sort(key=lambda entry: (entry.category, entry.name))
    return SourceRegistry(root=sources_root, entries=tuple(entries))


def write_registry(
    root: str | Path = DEFAULT_SOURCES_ROOT, output_path: str | Path | None = None
) -> SourceRegistry:
    sources_root = Path(root)
    registry = build_registry(sources_root)
    target = Path(output_path) if output_path is not None else sources_root / "registry.yaml"
    target.write_text(yaml.safe_dump(registry.to_dict(), sort_keys=False), encoding="utf-8")
    return registry
