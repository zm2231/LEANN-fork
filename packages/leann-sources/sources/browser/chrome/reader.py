"""Chrome history source-registry reader."""

from __future__ import annotations

from leann_sources.base import SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.documents import DocumentSourceReader


class ChromeSourceReader(DocumentSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    @property
    def history_path(self):
        return self.path / "History"

    def validate(self) -> ValidationReport:
        if not self.history_path.exists():
            return ValidationReport(
                False,
                "missing data",
                errors=[f"{self.history_path} does not exist"],
            )
        return ValidationReport(True, "ok")

    def load_documents(self, *, max_count: int = -1):
        from apps.history_data.history import ChromeHistoryReader

        return ChromeHistoryReader().load_data(
            chrome_profile_path=str(self.path),
            max_count=max_count,
        )

    def stats(self) -> SourceStats:
        if not self.history_path.exists():
            return SourceStats()
        return SourceStats(count=len(self.load_documents(max_count=-1)))
