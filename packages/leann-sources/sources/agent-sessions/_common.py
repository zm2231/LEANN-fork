"""Shared helpers for agent-session readers.

All three agent-session readers (claude-code, codex, pi-agent) ingest local
JSONL files and emit SIGNALS-compliant chunks. They share helpers for:

- UTC ISO timestamp normalization (`to_utc_iso`)
- Path-string project_id resolution without touching filesystem state
  (`resolve_project_id`)
- Local user identity resolution via env var (`resolve_local_user`)
- Long tool-output truncation with provenance flag (`truncate`)
- Portable source_document_id formatting (`document_id`)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

_HOME = Path.home()
_GENERIC_BASENAMES = frozenset(
    {"workspace", "src", "lib", "code", "home", "work", "tmp", "var"}
)


def to_utc_iso(ts: str | None) -> str | None:
    """Parse a timestamp string and return UTC ISO 8601 with offset, or None.

    Accepts ``Z``-suffixed ISO strings as well as fully-qualified offsets.
    Naive datetimes are forced to UTC. Malformed input returns ``None`` so
    callers can skip the chunk instead of emitting non-conformant metadata.
    """
    if not ts:
        return None
    text = str(ts).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.isoformat()


def resolve_project_id(cwd: str | None) -> str:
    """Derive a project_id from a cwd path string. Filesystem state is NOT consulted.

    Deterministic by design: the same cwd always yields the same id, even if the
    directory is later renamed, deleted, or moved. ``extra.cwd`` still carries the
    unambiguous full path.

    Rules:
    - empty/None cwd → ``"unknown"``
    - cwd == $HOME → ``"home"``
    - basename is generic (``workspace``, ``src``, ``lib``, ``code``, ...)
      → last two path components joined with ``/``
    - otherwise → basename
    """
    if not cwd:
        return "unknown"
    p = Path(cwd)
    if p == _HOME:
        return "home"
    parts = [part for part in p.parts if part not in {"/", ""}]
    if not parts:
        return "root"
    last = parts[-1]
    if last in _GENERIC_BASENAMES and len(parts) >= 2:
        return f"{parts[-2]}/{last}"
    return last


def resolve_local_user(default: str = "local_user") -> str:
    """Return the local user identity for ``author`` on user-authored chunks.

    Reads ``LEANN_LOCAL_USER`` from the environment; falls back to ``default``.
    Never derives from ``$USER`` or other host-leaking sources.
    """
    value = os.environ.get("LEANN_LOCAL_USER", "").strip()
    return value or default


def truncate(text: str, limit: int) -> tuple[str, bool, int]:
    """Truncate ``text`` to ``limit`` characters. Returns (text, was_truncated, original_length).

    Caller can record ``extra.truncated=True`` and ``extra.original_length`` to
    preserve provenance.
    """
    original_length = len(text)
    if original_length <= limit:
        return text, False, original_length
    return text[:limit], True, original_length


def document_id(agent: str, session_id: str) -> str:
    """Stable, portable source_document_id that does not leak local paths."""
    return f"agent:{agent}:session:{session_id}"


def resolve_extra_roots(env_var: str) -> list[str]:
    """Parse a comma- or colon-separated list of extra session roots from env."""
    raw = os.environ.get(env_var, "")
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.replace(":", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts
