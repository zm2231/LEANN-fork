"""Codex on-disk session reader (`$CODEX_HOME/sessions/**/rollout-*.{jsonl,json}`)."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (  # noqa: E402
    document_id,
    resolve_local_user,
    resolve_project_id,
    to_utc_iso,
    truncate,
)

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport  # noqa: E402
from leann_sources.manifest import SourceManifest  # noqa: E402
from leann_sources.readers._mapping import ManifestMappingMixin  # noqa: E402

_INDEXABLE_PAYLOAD_TYPES = {
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "tool_search_call",
    "tool_search_output",
}
_AGENT_NAME = "codex"
_META_SCAN_LIMIT = 10
_MAX_TOOL_OUTPUT_CHARS = 4000


class CodexSourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, root: str | Path | None = None):
        self.manifest = manifest
        configured = root or os.environ.get("CODEX_HOME") or manifest.data.get(
            "default_path", "~/.codex/sessions"
        )
        path = Path(os.path.expanduser(str(configured)))
        if path.name != "sessions" and (path / "sessions").exists():
            path = path / "sessions"
        self.root = path
        self._local_user = resolve_local_user()

    def discover(self) -> DiscoveryResult:
        return DiscoveryResult(
            found=self.root.exists(),
            checked=[str(self.root)],
            paths=[self.root] if self.root.exists() else [],
            missing=[] if self.root.exists() else [str(self.root)],
        )

    def validate(self) -> ValidationReport:
        if not self.root.exists():
            return ValidationReport(
                False, "missing data", errors=[f"{self.root} does not exist"]
            )
        if not self.root.is_dir():
            return ValidationReport(
                False, "not a directory", errors=[f"{self.root} is not a directory"]
            )
        return ValidationReport(True, "ok")

    def stats(self) -> SourceStats:
        return SourceStats(count=sum(1 for _ in self._iter_session_files()))

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]:
        document_counts: defaultdict[Any, int] = defaultdict(int)
        for session_file in self._iter_session_files():
            file_mtime = datetime.fromtimestamp(session_file.stat().st_mtime, tz=UTC)
            if since is not None and file_mtime < since:
                continue
            yield from self._iter_session_chunks(session_file, file_mtime, document_counts)

    def _iter_session_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("rollout-*")):
            if path.is_file() and path.suffix in {".jsonl", ".json"}:
                yield path

    def _iter_session_chunks(
        self,
        session_file: Path,
        file_mtime: datetime,
        document_counts: defaultdict[Any, int],
    ) -> Iterator[Chunk]:
        modified_at_iso = to_utc_iso(file_mtime.isoformat())
        meta = _load_session_meta(session_file)
        session_id = meta.get("id", session_file.stem)
        session_started_at = to_utc_iso(meta.get("timestamp"))
        cwd = meta.get("cwd")
        project_id = resolve_project_id(cwd)
        doc_id = document_id(_AGENT_NAME, session_id)
        cli_version = meta.get("cli_version")
        originator = meta.get("originator")
        row_index = 0
        with session_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "response_item":
                    continue
                payload = event.get("payload") or {}
                payload_type = payload.get("type")
                if payload_type not in _INDEXABLE_PAYLOAD_TYPES:
                    continue
                raw_text = _extract_text(payload_type, payload)
                if not raw_text:
                    continue
                text, truncated, original_length = truncate(raw_text, _MAX_TOOL_OUTPUT_CHARS)
                event_time = to_utc_iso(event.get("timestamp")) or session_started_at
                if not event_time:
                    continue
                role = payload.get("role") if payload_type == "message" else None
                author = self._local_user if role == "user" else _AGENT_NAME
                event_id = payload.get("id") or payload.get("call_id") or str(row_index)
                source_id = f"{session_id}:{payload_type}:{event_id}"
                record = {
                    "text": text,
                    "created_at": event_time,
                    "modified_at": modified_at_iso,
                    "event_time": event_time,
                    "author": author,
                    "activity_type": "authored" if payload_type in {"message", "reasoning"} else "created",
                    "participant_ids": [self._local_user, _AGENT_NAME],
                    "source_id": source_id,
                    "source_document_id": doc_id,
                    "project_id": project_id,
                    "parent_ref": f"session:{session_id}",
                    "extra": {
                        "session_id": session_id,
                        "session_started_at": session_started_at,
                        "session_path": str(session_file),
                        "cwd": cwd,
                        "role": role,
                        "payload_type": payload_type,
                        "truncated": truncated,
                        "original_length": original_length,
                        "cli_version": cli_version,
                        "originator": originator,
                        "provider": "openai",
                        "agent": "codex",
                    },
                }
                chunk = self._chunk_from_record(
                    record, row_index=row_index, text=text, document_counts=document_counts
                )
                chunk.metadata["extra"] = record["extra"]
                yield chunk
                row_index += 1


def _load_session_meta(session_file: Path) -> dict[str, Any]:
    try:
        with session_file.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                if i >= _META_SCAN_LIMIT:
                    return {}
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "session_meta":
                    payload = event.get("payload") or {}
                    return payload if isinstance(payload, dict) else {}
    except OSError:
        return {}
    return {}


def _extract_text(payload_type: str, payload: dict[str, Any]) -> str:
    if payload_type == "message":
        content = payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in {"input_text", "output_text", "text"}:
                    parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p)
        return ""
    if payload_type == "reasoning":
        summary = payload.get("summary")
        if isinstance(summary, list):
            parts = []
            for block in summary:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
            return "[reasoning] " + "\n".join(p for p in parts if p)
        if isinstance(summary, str):
            return f"[reasoning] {summary}"
        encrypted = payload.get("encrypted_content")
        return "[reasoning] (encrypted)" if encrypted else ""
    if payload_type == "function_call":
        name = payload.get("name", "")
        arguments = payload.get("arguments", "")
        return f"[function_call:{name}] {arguments}"
    if payload_type == "function_call_output":
        output = payload.get("output", "")
        if isinstance(output, dict):
            output = json.dumps(output)
        return f"[function_call_output] {output}"
    if payload_type == "tool_search_call":
        query = payload.get("query", "")
        return f"[tool_search_call] {query}"
    if payload_type == "tool_search_output":
        results = payload.get("results", "")
        if isinstance(results, list):
            results = json.dumps(results)
        return f"[tool_search_output] {results}"
    return ""
