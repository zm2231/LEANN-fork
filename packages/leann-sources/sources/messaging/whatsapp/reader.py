"""WhatsApp ChatStorage.sqlite source-registry reader."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime

from leann_sources.base import Chunk, ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.sqlite import SQLiteSourceReader


class WhatsAppReader(SQLiteSourceReader):
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

        required = {"ZWAMESSAGE", "ZWACHATSESSION"}
        missing = sorted(required - tables)
        if missing:
            return ValidationReport(
                False,
                "missing schema",
                errors=[f"missing WhatsApp tables: {', '.join(missing)}"],
            )
        return ValidationReport(True, "ok")

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts = defaultdict(int)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(self._query(connection), {"since": since})
            for row_index, row in enumerate(rows):
                record = dict(row)
                record["modified_date"] = record["edited_date"] or record["message_date"]
                chunk = self._chunk_from_record(
                    record, row_index=row_index, document_counts=document_counts
                )
                if not record["edited_date"]:
                    chunk.metadata["modified_at_synthesized"] = True
                yield chunk

    def _query(self, connection: sqlite3.Connection) -> str:
        edited_date_expr = (
            "m.ZEDITEDDATE" if self._has_column(connection, "ZWAMESSAGE", "ZEDITEDDATE") else "0"
        )
        from_jid_expr = (
            "m.ZFROMJID" if self._has_column(connection, "ZWAMESSAGE", "ZFROMJID") else "NULL"
        )
        contact_jid_expr = (
            "m.ZCONTACTJID" if self._has_column(connection, "ZWAMESSAGE", "ZCONTACTJID") else "NULL"
        )
        return f"""
            SELECT
              m.Z_PK AS message_id,
              m.ZTEXT AS text,
              m.ZMESSAGEDATE AS message_date,
              {edited_date_expr} AS edited_date,
              m.ZISFROMME AS is_from_me,
              COALESCE({from_jid_expr}, '') AS from_jid,
              COALESCE({contact_jid_expr}, '') AS contact_jid,
              COALESCE(c.ZCONTACTJID, {contact_jid_expr}, {from_jid_expr}, 'unknown') AS chat_jid,
              COALESCE(c.ZPARTNERNAME, c.ZGROUPNAME, c.ZCONTACTJID, 'Unknown Chat') AS chat_name,
              ('whatsapp_chat:' || COALESCE(c.Z_PK, m.ZCHATSESSION, 0)) AS source_document_id,
              ('whatsapp_message:' || m.Z_PK) AS source_id,
              CASE
                WHEN m.ZISFROMME = 1 THEN 'You'
                ELSE COALESCE({from_jid_expr}, {contact_jid_expr}, 'Unknown')
              END AS sender,
              COALESCE({from_jid_expr}, {contact_jid_expr}, c.ZCONTACTJID, '') AS participant_handles
            FROM ZWAMESSAGE m
            LEFT JOIN ZWACHATSESSION c ON m.ZCHATSESSION = c.Z_PK
            WHERE m.ZTEXT IS NOT NULL AND m.ZTEXT != ''
            ORDER BY c.Z_PK, m.ZMESSAGEDATE
        """

    def _has_column(self, connection: sqlite3.Connection, table: str, column: str) -> bool:
        return column in {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
