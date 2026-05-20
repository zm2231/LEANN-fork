"""API source reader base class."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import requests

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers._mapping import ManifestMappingMixin


class APISourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, session: requests.Session | None = None):
        self.manifest = manifest
        self.session = session or requests.Session()
        self.rate_limit_remaining: int | None = None

    def discover(self) -> DiscoveryResult:
        base_url = self.manifest.data.get("base_url")
        return DiscoveryResult(found=bool(base_url), checked=[str(base_url)] if base_url else [])

    def validate(self) -> ValidationReport:
        missing_env = [
            env_var
            for env_var in (self.manifest.auth or {}).get("env_vars", [])
            if not os.environ.get(env_var)
        ]
        if missing_env:
            return ValidationReport(
                False,
                "missing auth",
                errors=[f"missing environment variable: {env_var}" for env_var in missing_env],
            )
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        row_index = 0
        document_counts = defaultdict(int)
        for record in self.iter_records(since=since):
            yield self._chunk_from_record(
                record, row_index=row_index, document_counts=document_counts
            )
            row_index += 1

    def iter_records(self, since: datetime | None = None) -> Iterator[dict[str, Any]]:
        fixture_pages = self.manifest.data.get("pages")
        if fixture_pages is not None:
            for page in fixture_pages:
                for record in page:
                    yield record
            return

        next_url = self.manifest.data.get("url") or self.manifest.data.get("base_url")
        while next_url:
            payload, headers = self._fetch_page(next_url)
            remaining = headers.get("X-RateLimit-Remaining")
            self.rate_limit_remaining = (
                int(remaining) if remaining and remaining.isdigit() else None
            )
            for record in payload.get(self.manifest.data.get("items_key", "items"), []):
                yield record
            next_url = payload.get(self.manifest.data.get("next_key", "next"))

    def stats(self) -> SourceStats:
        fixture_pages = self.manifest.data.get("pages") or []
        return SourceStats(count=sum(len(page) for page in fixture_pages))

    def _fetch_page(self, url: str) -> tuple[dict[str, Any], dict[str, str]]:
        response = self.session.get(url, timeout=self.manifest.data.get("timeout", 30))
        response.raise_for_status()
        return response.json(), dict(response.headers)
