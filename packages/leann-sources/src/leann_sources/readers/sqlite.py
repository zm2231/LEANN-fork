"""SQLite source reader base class."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers._mapping import ManifestMappingMixin


class SQLiteSourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, path: str | Path | None = None):
        self.manifest = manifest
        self.path = Path(path or manifest.data.get("default_path", "")).expanduser()

    def discover(self) -> DiscoveryResult:
        return DiscoveryResult(
            found=self.path.exists(),
            checked=[str(self.path)],
            paths=[self.path] if self.path.exists() else [],
            missing=[] if self.path.exists() else [str(self.path)],
        )

    def validate(self) -> ValidationReport:
        if not self.path.exists():
            return ValidationReport(False, "missing data", errors=[f"{self.path} does not exist"])
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("select 1")
        except sqlite3.Error as exc:
            return ValidationReport(False, "sqlite error", errors=[str(exc)])
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        query = self.manifest.data.get("query")
        if not query:
            raise ValueError("SQLite source manifests must declare data.query")
        document_counts = defaultdict(int)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, {"since": since.isoformat() if since else None})
            for row_index, row in enumerate(rows):
                yield self._chunk_from_record(
                    dict(row), row_index=row_index, document_counts=document_counts
                )

    def stats(self) -> SourceStats:
        if not self.path.exists():
            return SourceStats()
        count_query = self.manifest.data.get("count_query")
        if not count_query:
            return SourceStats()
        with sqlite3.connect(self.path) as connection:
            count = int(connection.execute(count_query).fetchone()[0])
        return SourceStats(count=count)
