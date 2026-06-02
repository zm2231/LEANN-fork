"""Pi agent on-disk session reader (multi-root JSONL tree)."""

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
    resolve_extra_roots,
    resolve_local_user,
    resolve_project_id,
    to_utc_iso,
)

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport  # noqa: E402
from leann_sources.manifest import SourceManifest  # noqa: E402
from leann_sources.readers._mapping import ManifestMappingMixin  # noqa: E402

_AGENT_NAME = "pi_agent"
_HEADER_SCAN_LIMIT = 10
_EXTRA_ROOTS_ENV = "LEANN_PI_AGENT_ROOTS"


class PiAgentSourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, roots: list[str | Path] | None = None):
        self.manifest = manifest
        configured: list[str | Path] = []
        if roots is not None:
            configured.extend(roots)
        else:
            configured.extend(manifest.data.get("roots", []))
            configured.extend(resolve_extra_roots(_EXTRA_ROOTS_ENV))
        if not configured:
            configured = [manifest.data.get("default_path", "~/.pi/agent/sessions")]
        seen_strs: set[str] = set()
        self.roots: list[Path] = []
        for raw in configured:
            text = str(raw).strip()
            if not text or text in seen_strs:
                continue
            seen_strs.add(text)
            self.roots.append(Path(os.path.expanduser(text)))
        self._local_user = resolve_local_user()

    def discover(self) -> DiscoveryResult:
        existing = [r for r in self.roots if r.exists()]
        missing = [str(r) for r in self.roots if not r.exists()]
        return DiscoveryResult(
            found=bool(existing),
            checked=[str(r) for r in self.roots],
            paths=existing,
            missing=missing,
        )

    def validate(self) -> ValidationReport:
        existing = [r for r in self.roots if r.exists()]
        if not existing:
            return ValidationReport(
                False,
                "missing data",
                errors=[f"none of the configured roots exist: {[str(r) for r in self.roots]}"],
            )
        for root in existing:
            if not root.is_dir():
                return ValidationReport(
                    False, "not a directory", errors=[f"{root} is not a directory"]
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
        seen: set[Path] = set()
        for root in self.roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.jsonl")):
                if not path.is_file():
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path

    def _iter_session_chunks(
        self,
        session_file: Path,
        file_mtime: datetime,
        document_counts: defaultdict[Any, int],
    ) -> Iterator[Chunk]:
        modified_at_iso = to_utc_iso(file_mtime.isoformat())
        header = _load_header(session_file)
        session_id = header.get("id", session_file.stem)
        session_started_at = to_utc_iso(header.get("timestamp"))
        cwd = header.get("cwd")
        project_id = resolve_project_id(cwd)
        doc_id = document_id(_AGENT_NAME, session_id)

        provider: str | None = None
        model_id: str | None = None
        thinking_level: str | None = None
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
                etype = event.get("type")
                if etype == "model_change":
                    provider = event.get("provider", provider)
                    model_id = event.get("modelId", model_id)
                    continue
                if etype == "thinking_level_change":
                    thinking_level = event.get("thinkingLevel", thinking_level)
                    continue

                role: str
                text: str
                if etype == "message":
                    message = event.get("message") or {}
                    text = _extract_message_text(message)
                    role = message.get("role", "assistant")
                elif etype == "custom_message":
                    if event.get("display") is False:
                        continue
                    text = _extract_custom_text(event)
                    role = "system"
                else:
                    continue
                if not text:
                    continue
                event_time = to_utc_iso(event.get("timestamp"))
                if not event_time:
                    continue
                author = self._local_user if role == "user" else _AGENT_NAME
                event_id = event.get("id", str(row_index))
                source_id = f"{session_id}:{etype}:{event_id}"
                record = {
                    "text": text,
                    "created_at": event_time,
                    "modified_at": modified_at_iso,
                    "event_time": event_time,
                    "author": author,
                    "activity_type": "authored" if role in {"user", "assistant"} else "created",
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
                        "event_type": etype,
                        "custom_type": event.get("customType") if etype == "custom_message" else None,
                        "parent_id": event.get("parentId"),
                        "provider": provider,
                        "model_id": model_id,
                        "thinking_level": thinking_level,
                        "agent": "pi-agent",
                    },
                }
                chunk = self._chunk_from_record(
                    record, row_index=row_index, text=text, document_counts=document_counts
                )
                chunk.metadata["extra"] = record["extra"]
                yield chunk
                row_index += 1


def _load_header(session_file: Path) -> dict[str, Any]:
    try:
        with session_file.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                if i >= _HEADER_SCAN_LIMIT:
                    return {}
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "session":
                    return event
    except OSError:
        return {}
    return {}


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "thinking":
                thinking = block.get("thinking") or block.get("text")
                if thinking:
                    parts.append(f"[thinking] {thinking}")
            elif block_type == "toolUse":
                name = block.get("name", "")
                parts.append(f"[tool_use:{name}]")
            elif block_type == "toolResult":
                result = block.get("content") or block.get("text")
                if result:
                    parts.append(f"[tool_result] {result}")
        return "\n".join(p for p in parts if p)
    return ""


def _extract_custom_text(event: dict[str, Any]) -> str:
    content = event.get("content")
    if isinstance(content, str):
        custom_type = event.get("customType", "")
        return f"[custom_message:{custom_type}] {content}" if custom_type else content
    return ""
