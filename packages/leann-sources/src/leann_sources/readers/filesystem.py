"""Filesystem source reader base class."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers._mapping import ManifestMappingMixin


class FilesystemSourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, root: str | Path | None = None):
        self.manifest = manifest
        self.root = Path(root or manifest.data.get("default_path", "")).expanduser()

    def discover(self) -> DiscoveryResult:
        return DiscoveryResult(
            found=self.root.exists(),
            checked=[str(self.root)],
            paths=[self.root] if self.root.exists() else [],
            missing=[] if self.root.exists() else [str(self.root)],
        )

    def validate(self) -> ValidationReport:
        if not self.root.exists():
            return ValidationReport(False, "missing data", errors=[f"{self.root} does not exist"])
        if not self.root.is_dir():
            return ValidationReport(
                False, "not a directory", errors=[f"{self.root} is not a directory"]
            )
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts = defaultdict(int)
        for row_index, path in enumerate(self._iter_files()):
            stat = path.stat()
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
            if since is not None and modified_at < since:
                continue
            text = path.read_text(encoding=self.manifest.data.get("encoding", "utf-8"))
            record = {
                "path": str(path),
                "relative_path": str(path.relative_to(self.root)),
                "name": path.name,
                "text": text,
                "stat.st_mtime": stat.st_mtime,
                "source_document_id": str(path),
            }
            if hasattr(stat, "st_birthtime"):
                record["stat.st_birthtime"] = stat.st_birthtime
            yield self._chunk_from_record(
                record, row_index=row_index, text=text, document_counts=document_counts
            )

    def stats(self) -> SourceStats:
        return SourceStats(count=sum(1 for _ in self._iter_files()))

    def _iter_files(self) -> Iterator[Path]:
        pattern = self.manifest.data.get("glob", "**/*")
        for path in sorted(self.root.glob(pattern)):
            if path.is_file():
                yield path
