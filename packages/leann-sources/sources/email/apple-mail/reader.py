"""Apple Mail source-registry reader."""

from __future__ import annotations

from leann_sources.base import SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.documents import DocumentSourceReader


class AppleMailReader(DocumentSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def _message_dirs(self):
        from apps.email_data.LEANN_email_reader import find_all_messages_directories

        return find_all_messages_directories(str(self.path))

    def validate(self) -> ValidationReport:
        if not self.path.exists():
            return ValidationReport(False, "missing data", errors=[f"{self.path} does not exist"])
        if not self._message_dirs():
            return ValidationReport(
                False,
                "missing data",
                errors=[f"no Apple Mail Messages directories under {self.path}"],
            )
        return ValidationReport(True, "ok")

    def load_documents(self, *, max_count: int = -1):
        from apps.email_data.LEANN_email_reader import EmlxReader

        reader = EmlxReader()
        docs = []
        remaining = max_count
        for message_dir in self._message_dirs():
            limit = remaining if remaining > 0 else max_count
            batch = reader.load_data(str(message_dir), max_count=limit)
            docs.extend(batch)
            if remaining > 0:
                remaining -= len(batch)
                if remaining <= 0:
                    break
        return docs

    def stats(self) -> SourceStats:
        return SourceStats(
            count=sum(len(list(path.glob("**/*.emlx"))) for path in self._message_dirs())
        )
