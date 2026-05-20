"""Adapters for existing document-style source readers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest


class DocumentSourceReader:
    def __init__(self, manifest: SourceManifest):
        self.manifest = manifest
        self.path = Path(manifest.data.get("default_path", "")).expanduser()

    def discover(self) -> DiscoveryResult:
        found = self.path.exists()
        return DiscoveryResult(
            found=found,
            checked=[str(self.path)],
            paths=[self.path] if found else [],
            missing=[] if found else [str(self.path)],
        )

    def validate(self) -> ValidationReport:
        if not self.path.exists():
            return ValidationReport(False, "missing data", errors=[f"{self.path} does not exist"])
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: Any | None = None) -> Iterator[Chunk]:
        for doc in self.load_documents(max_count=-1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            yield Chunk(text=getattr(doc, "text", ""), metadata=metadata)

    def stats(self) -> SourceStats:
        return SourceStats(count=len(list(self.iter_chunks())))

    def load_documents(self, *, max_count: int = -1) -> list[Any]:
        raise NotImplementedError
