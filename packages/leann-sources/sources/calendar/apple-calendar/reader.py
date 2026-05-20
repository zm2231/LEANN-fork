"""Apple Calendar source-registry reader."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from leann_sources.base import Chunk, SourceStats, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.sqlite import SQLiteSourceReader


class AppleCalendarReader(SQLiteSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def validate(self) -> ValidationReport:
        if not self.path.exists():
            return ValidationReport(False, "missing data", errors=[f"{self.path} does not exist"])
        try:
            with self._calendar_connection() as connection:
                connection.execute("select 1 from CI_EVENT limit 1")
        except sqlite3.Error as exc:
            return ValidationReport(False, "sqlite error", errors=[str(exc)])
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts = defaultdict(int)
        with self._calendar_connection() as connection:
            connection.row_factory = sqlite3.Row
            columns = {row[1] for row in connection.execute("PRAGMA table_info(CI_EVENT)")}
            rows = connection.execute(
                self._query(columns), (self.manifest.data.get("max_count", 1000),)
            )
            for row_index, row in enumerate(rows):
                record = dict(row)
                record["event_time"] = self._utc_iso(record["event_time"])
                record["event_time_local"] = self._local_iso(record["event_time_local"])
                record["end_time_local"] = self._local_iso(record["end_time_local"])
                created_at = self._utc_iso(record.get("created_at"))
                modified_at = self._utc_iso(record.get("modified_at"))
                record["created_at"] = created_at or record["event_time"]
                record["modified_at"] = modified_at or record["event_time"]
                chunk = self._chunk_from_record(
                    record,
                    row_index=row_index,
                    text=self._render_calendar_text(record),
                    document_counts=document_counts,
                )
                chunk.metadata["created_at_synthesized"] = created_at is None
                chunk.metadata["modified_at_synthesized"] = modified_at is None
                yield chunk

    def stats(self) -> SourceStats:
        if not self.path.exists():
            return SourceStats()
        with self._calendar_connection() as connection:
            count = int(connection.execute("select count(*) from CI_EVENT").fetchone()[0])
        return SourceStats(count=count)

    @contextmanager
    def _calendar_connection(self) -> Iterator[sqlite3.Connection]:
        if not self.path.exists():
            raise sqlite3.OperationalError(f"{self.path} does not exist")
        temp_file = tempfile.NamedTemporaryFile(prefix="leann_calendar_", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        shutil.copy2(self.path, temp_path)
        connection = sqlite3.connect(temp_path)
        try:
            yield connection
        finally:
            connection.close()
            temp_path.unlink(missing_ok=True)

    def _query(self, columns: set[str]) -> str:
        created_at_expr = self._timestamp_expr(
            columns, ("created_date", "creation_date", "date_created", "created", "creation_time")
        )
        modified_at_expr = self._timestamp_expr(
            columns,
            (
                "last_modified_date",
                "last_modified",
                "modified_date",
                "date_modified",
                "updated_date",
                "updated",
            ),
        )
        return f"""
            SELECT rowid AS event_id, summary, description, location,
                   datetime(start_date + 978307200, 'unixepoch') AS event_time,
                   datetime(start_date + 978307200, 'unixepoch', 'localtime') AS event_time_local,
                   datetime(end_date + 978307200, 'unixepoch', 'localtime') AS end_time_local,
                   {created_at_expr} AS created_at,
                   {modified_at_expr} AS modified_at
            FROM CI_EVENT
            WHERE summary IS NOT NULL AND summary != ''
            ORDER BY start_date DESC
            LIMIT ?
        """

    def _timestamp_expr(self, columns: set[str], candidates: tuple[str, ...]) -> str:
        for column in candidates:
            if column in columns:
                return f"datetime({column} + 978307200, 'unixepoch')"
        return "NULL"

    def _utc_iso(self, value: str | None) -> str | None:
        if not value:
            return None
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).isoformat()

    def _local_iso(self, value: str | None) -> str | None:
        if not value:
            return None
        return datetime.fromisoformat(value).astimezone().isoformat()

    def _render_calendar_text(self, record: dict) -> str:
        return (
            f"Event: {record['summary']}\n"
            f"Start: {record['event_time_local']}\n"
            f"End: {record['end_time_local']}\n"
            f"Location: {record.get('location') or ''}\n"
            f"Description: {record.get('description') or ''}"
        )
