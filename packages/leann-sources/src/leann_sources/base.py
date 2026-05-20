"""Base source-reader types for manifest-driven ingest."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DiscoveryResult:
    found: bool
    checked: list[str] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceStats:
    count: int = 0
    date_range: tuple[datetime | None, datetime | None] = (None, None)
    top_values: dict[str, list[tuple[str, int]]] = field(default_factory=dict)


class SourceReader(Protocol):
    manifest: Any

    def discover(self) -> DiscoveryResult:
        """Find raw data paths or URLs for this source."""

    def validate(self) -> ValidationReport:
        """Check accessibility, permissions, credentials, and schema compatibility."""

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        """Yield SIGNALS-compliant chunks, optionally limited to newer source data."""

    def stats(self) -> SourceStats:
        """Return source counts and summary values for CLI display."""
