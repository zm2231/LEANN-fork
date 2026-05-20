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
    sources_root: Path = DEFAULT_SOURCES_ROOT
    registry_path: Path | None = None
    cli_subcommands: tuple[Any, ...] = ()
    index_source_handler: Any | None = None

    def __post_init__(self) -> None:
        if self.registry_path is None:
            object.__setattr__(self, "registry_path", self.sources_root / "registry.yaml")

    def register_cli(self, subparsers: Any, core_cli: Any) -> None:
        from leann_sources.cli import SourceCLI

        SourceCLI(self.sources_root).register(subparsers)

    def handle_cli(self, args: Any, core_cli: Any) -> bool:
        from leann_sources.cli import SourceCLI

        return SourceCLI(self.sources_root).handle(args, core_cli=core_cli)


def get_plugin() -> SourcesPlugin:
    return SourcesPlugin()
