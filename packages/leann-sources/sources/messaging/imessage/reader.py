"""iMessage source-registry reader."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime

from leann_sources.base import Chunk, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.sqlite import SQLiteSourceReader


class IMessageReader(SQLiteSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def validate(self) -> ValidationReport:
        base_report = super().validate()
        if not base_report.ok:
            return base_report
        try:
            with sqlite3.connect(self.path) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "select name from sqlite_master where type = 'table'"
                    )
                }
        except sqlite3.Error as exc:
            return ValidationReport(False, "sqlite error", errors=[str(exc)])

        required = {"message", "chat", "handle", "chat_message_join"}
        missing = sorted(required - tables)
        if missing:
            return ValidationReport(
                False,
                "missing schema",
                errors=[f"missing iMessage tables: {', '.join(missing)}"],
            )
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts = defaultdict(int)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(self._query(connection), {"since": since})
            for row_index, row in enumerate(rows):
                yield self._chunk_from_record(
                    dict(row), row_index=row_index, document_counts=document_counts
                )

    def _query(self, connection: sqlite3.Connection) -> str:
        date_edited_expr = (
            "m.date_edited" if self._has_message_column(connection, "date_edited") else "0"
        )
        return f"""
            SELECT
              m.ROWID AS message_id,
              m.text,
              m.date,
              {date_edited_expr} AS date_edited,
              m.is_from_me,
              COALESCE(m.service, 'iMessage') AS service,
              COALESCE(c.chat_identifier, 'Unknown') AS chat_identifier,
              COALESCE(c.display_name, 'Unknown Chat') AS chat_display_name,
              COALESCE(h.id, 'Unknown') AS handle_id,
              COALESCE(c.ROWID, 0) AS chat_id,
              ('chat:' || COALESCE(c.ROWID, 0)) AS source_document_id,
              ('message:' || m.ROWID) AS source_id,
              CASE WHEN m.is_from_me = 1 THEN 'You' ELSE COALESCE(h.id, 'Unknown') END AS sender,
              COALESCE(h.id, '') AS participant_handles
            FROM message m
            LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
            LEFT JOIN chat c ON cmj.chat_id = c.ROWID
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            WHERE m.text IS NOT NULL AND m.text != ''
            ORDER BY c.ROWID, m.date
        """

    def _has_message_column(self, connection: sqlite3.Connection, column: str) -> bool:
        return column in {row[1] for row in connection.execute("PRAGMA table_info(message)")}
