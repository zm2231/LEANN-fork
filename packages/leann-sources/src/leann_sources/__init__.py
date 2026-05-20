"""LEANN source registry package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leann_sources.base import Chunk, SourceReader
from leann_sources.manifest import SourceManifest
from leann_sources.registry import DEFAULT_SOURCES_ROOT

__all__ = ["Chunk", "SourceManifest", "SourceReader", "get_plugin"]


@dataclass(frozen=True)
class SourcesPlugin:
    """Minimal plugin descriptor; Atom 2 fills in the callable hooks."""

    name: str = "sources"
    version: str = "0.1.0"
    registry_path: Path = DEFAULT_SOURCES_ROOT / "registry.yaml"
    cli_subcommands: tuple[Any, ...] = ()
    index_source_handler: Any | None = None


def get_plugin() -> SourcesPlugin:
    return SourcesPlugin()
