"""Export-zip source reader base class."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers._mapping import ManifestMappingMixin


class ExportZipSourceReader(ManifestMappingMixin):
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
        if not zipfile.is_zipfile(self.path):
            return ValidationReport(
                False, "not a zip file", errors=[f"{self.path} is not a zip file"]
            )
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts = defaultdict(int)
        for row_index, record in enumerate(self.iter_records()):
            yield self._chunk_from_record(
                record, row_index=row_index, document_counts=document_counts
            )

    def iter_records(self) -> Iterator[dict[str, Any]]:
        with zipfile.ZipFile(self.path) as archive:
            for name in sorted(archive.namelist()):
                if not name.endswith(".json"):
                    continue
                payload = json.loads(archive.read(name).decode("utf-8"))
                for record in self._records_from_payload(payload):
                    if "source_document_id" not in record:
                        record["source_document_id"] = name
                    yield record

    def stats(self) -> SourceStats:
        return SourceStats(count=sum(1 for _ in self.iter_records()) if self.path.exists() else 0)

    def _records_from_payload(self, payload: Any) -> Iterator[dict[str, Any]]:
        items_key = self.manifest.data.get("items_key")
        if items_key:
            payload = payload.get(items_key, [])
        if isinstance(payload, dict):
            yield payload
        else:
            yield from payload
