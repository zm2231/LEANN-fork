"""Claude Code on-disk session reader (`~/.claude/projects/<encoded-cwd>/<session>.jsonl`)."""

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
)

from leann_sources.base import Chunk, DiscoveryResult, SourceStats, ValidationReport  # noqa: E402
from leann_sources.manifest import SourceManifest  # noqa: E402
from leann_sources.readers._mapping import ManifestMappingMixin  # noqa: E402

_MESSAGE_TYPES = {"user", "assistant"}
_AGENT_NAME = "claude_code"
_CWD_SCAN_LIMIT = 200


def _decode_dir_cwd(dir_name: str) -> str:
    if dir_name.startswith("-"):
        return "/" + dir_name[1:].replace("-", "/")
    return dir_name


class ClaudeCodeSourceReader(ManifestMappingMixin):
    def __init__(self, manifest: SourceManifest, root: str | Path | None = None):
        self.manifest = manifest
        self.root = Path(
            root or os.path.expanduser(manifest.data.get("default_path", "~/.claude/projects"))
        )
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
        for project_dir in sorted(self.root.iterdir()):
            if not project_dir.is_dir():
                continue
            for session_file in sorted(project_dir.glob("*.jsonl")):
                yield session_file

    def _iter_session_chunks(
        self,
        session_file: Path,
        file_mtime: datetime,
        document_counts: defaultdict[Any, int],
    ) -> Iterator[Chunk]:
        session_id = session_file.stem
        modified_at_iso = to_utc_iso(file_mtime.isoformat())
        doc_id = document_id(_AGENT_NAME, session_id)
        cwd_from_events = self._scan_cwd(session_file)
        cwd = cwd_from_events or _decode_dir_cwd(session_file.parent.name)
        project_id = resolve_project_id(cwd)
        session_started_at: str | None = None
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
                event_type = event.get("type")
                if event_type not in _MESSAGE_TYPES:
                    continue
                if event.get("isMeta"):
                    continue
                text = _extract_text(event)
                if not text:
                    continue
                event_time = to_utc_iso(event.get("timestamp"))
                if not event_time:
                    continue
                if session_started_at is None:
                    session_started_at = event_time
                role = event.get("message", {}).get("role", event_type)
                author = self._local_user if role == "user" else _AGENT_NAME
                uuid = event.get("uuid", "")
                source_id = f"{session_id}:{uuid}" if uuid else f"{session_id}:{row_index}"
                record = {
                    "text": text,
                    "created_at": event_time,
                    "modified_at": modified_at_iso,
                    "event_time": event_time,
                    "author": author,
                    "activity_type": "authored",
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
                        "uuid": uuid,
                        "parent_uuid": event.get("parentUuid"),
                        "is_sidechain": event.get("isSidechain", False),
                        "model": event.get("message", {}).get("model"),
                        "provider": "anthropic",
                        "agent": "claude-code",
                    },
                }
                chunk = self._chunk_from_record(
                    record, row_index=row_index, text=text, document_counts=document_counts
                )
                chunk.metadata["extra"] = record["extra"]
                yield chunk
                row_index += 1

    @staticmethod
    def _scan_cwd(session_file: Path) -> str | None:
        try:
            with session_file.open("r", encoding="utf-8", errors="replace") as handle:
                for i, line in enumerate(handle):
                    if i >= _CWD_SCAN_LIMIT:
                        return None
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cwd = event.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
        except OSError:
            return None
        return None


def _extract_text(event: dict[str, Any]) -> str:
    message = event.get("message")
    if not isinstance(message, dict):
        return ""
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
                thinking = block.get("thinking")
                if thinking:
                    parts.append(f"[thinking] {thinking}")
            elif block_type == "tool_use":
                name = block.get("name", "")
                parts.append(f"[tool_use:{name}]")
            elif block_type == "tool_result":
                result = block.get("content")
                if isinstance(result, str):
                    parts.append(f"[tool_result] {result}")
                elif isinstance(result, list):
                    for sub in result:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(f"[tool_result] {sub.get('text', '')}")
        return "\n".join(p for p in parts if p)
    return ""
