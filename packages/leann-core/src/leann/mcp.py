#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
import subprocess
import sys
from typing import Any

_base_dir: str | None = None


@contextlib.contextmanager
def _mcp_cwd():
    if not _base_dir:
        yield
        return
    old = os.getcwd()
    os.chdir(_base_dir)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def _suppress_stdout_fd():
    """Suppress Python and native stdout while preserving JSON-RPC stdio framing."""
    sys.stdout.flush()
    saved_fd = None
    devnull_file = None
    try:
        saved_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.close(devnull)
        devnull_file = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull_file):
            yield
    finally:
        sys.stdout.flush()
        if devnull_file is not None:
            devnull_file.close()
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)


def _parse_json_object(value: Any, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if value in (None, ""):
        return None, None
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            return None, f"Error: {label} is not valid JSON: {e}"
        if isinstance(parsed, dict):
            return parsed, None
    return None, f"Error: {label} must be a JSON object"


def _result_to_dict(result, *, query: str | None = None, rank: int | None = None) -> dict[str, Any]:
    out = {
        "id": result.id,
        "score": float(result.score),
        "text": result.text,
        "metadata": result.metadata,
    }
    if query is not None:
        out["query"] = query
    if rank is not None:
        out["rank"] = rank
    siblings = getattr(result, "siblings", None)
    if siblings:
        out["siblings"] = [_result_to_dict(s) for s in siblings]
    return out


def _rrf_fuse(
    result_lists: list[list],
    queries: list[str],
    *,
    limit: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    order = 0
    for query, results in zip(queries, result_lists):
        for rank, result in enumerate(results, 1):
            key = str(result.id)
            entry = fused.get(key)
            if entry is None:
                entry = {
                    "result": result,
                    "rrf_score": 0.0,
                    "best_rank": rank,
                    "matched_queries": [],
                    "_order": order,
                }
                fused[key] = entry
                order += 1
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
            entry["best_rank"] = min(entry["best_rank"], rank)
            if query not in entry["matched_queries"]:
                entry["matched_queries"].append(query)

    ranked = sorted(
        fused.values(),
        key=lambda e: (-e["rrf_score"], e["best_rank"], e["_order"]),
    )[:limit]
    output = []
    for entry in ranked:
        row = _result_to_dict(entry["result"])
        row["rrf_score"] = entry["rrf_score"]
        row["best_rank"] = entry["best_rank"]
        row["matched_queries"] = entry["matched_queries"]
        output.append(row)
    return output


def _resolve_index_path(index_name: str) -> str:
    from .cli import LeannCLI

    cli = LeannCLI()
    resolved = cli._resolve_index_path(index_name, non_interactive=True, purpose="search", quiet=True)
    if not resolved:
        raise ValueError(f"Index not found: {index_name}")
    return resolved


def _direct_multi_search(args: dict[str, Any]) -> dict[str, Any]:
    from .api import LeannSearcher

    if not args.get("index_name") or not args.get("query"):
        raise ValueError("index_name and query are required")

    query = str(args["query"])
    extra_queries = args.get("extra_queries") or args.get("extraQueries") or []
    if isinstance(extra_queries, str):
        extra_queries = [extra_queries]
    if not isinstance(extra_queries, list):
        raise ValueError("extra_queries must be a list of strings")
    queries = [query] + [str(q) for q in extra_queries if str(q).strip()]

    search_mode = args.get("search_mode") or args.get("searchMode") or "prose"
    if search_mode not in {"prose", "code", "exact", "filtered"}:
        raise ValueError("search_mode must be one of: prose, code, exact, filtered")

    default_top_k = 12 if search_mode in {"prose", "filtered"} else 5
    default_fetch = 50 if search_mode == "filtered" else 30 if len(queries) > 1 else default_top_k
    top_k = int(args.get("top_k", args.get("topK", default_top_k)))
    fetch = int(args.get("fetch", max(top_k, default_fetch)))
    limit = int(args.get("limit", top_k))
    complexity = int(args.get("complexity", 64))

    default_weight = {
        "prose": 0.3,
        "code": 1.0,
        "exact": 0.0,
        "filtered": 0.3,
    }[search_mode]
    vector_weight = float(args.get("vector_weight", args.get("vectorWeight", default_weight)))

    metadata_filters, err = _parse_json_object(
        args.get("metadata_filters", args.get("metadataFilters")), "metadata_filters"
    )
    if err:
        raise ValueError(err)

    search_kwargs: dict[str, Any] = {
        "complexity": complexity,
        "metadata_filters": metadata_filters,
        "vector_weight": vector_weight,
        "prefilter": args.get("prefilter", "auto"),
    }
    for src, dest, cast in (
        ("diversify_by", "diversify_by", str),
        ("diversifyBy", "diversify_by", str),
        ("max_per_group", "max_per_group", int),
        ("maxPerGroup", "max_per_group", int),
        ("context_window", "context_window", int),
        ("contextWindow", "context_window", int),
    ):
        if src in args:
            search_kwargs[dest] = cast(args[src])

    with _mcp_cwd():
        index_path = _resolve_index_path(str(args["index_name"]))
    with _suppress_stdout_fd():
        with LeannSearcher(index_path=index_path, enable_warmup=False) as searcher:
            result_lists = searcher.multi_search(queries, top_k=fetch, **search_kwargs)
    plain_lists = [item[0] if isinstance(item, tuple) else item for item in result_lists]
    fused = _rrf_fuse(plain_lists, queries, limit=limit)
    return {
        "index_name": args["index_name"],
        "search_mode": search_mode,
        "queries": queries,
        "vector_weight": vector_weight,
        "fetch": fetch,
        "limit": limit,
        "results": fused,
    }


def _leann_cmd() -> list[str]:
    """Build the base command for invoking ``leann`` CLI.

    Using ``sys.executable -m leann`` guarantees we find the CLI even when
    the ``leann`` console-script is not on PATH (common on Windows when
    launched by mcp-proxy or other service wrappers).
    """
    return [sys.executable, "-m", "leann"]


def handle_request(request):
    if request.get("method") == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "capabilities": {"tools": {}},
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "leann-mcp", "version": "1.0.0"},
            },
        }

    elif request.get("method") == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "leann_search",
                        "description": """🔍 Search code using natural language - like having a coding assistant who knows your entire codebase!

🎯 **Perfect for**:
- "How does authentication work?" → finds auth-related code
- "Error handling patterns" → locates try-catch blocks and error logic
- "Database connection setup" → finds DB initialization code
- "API endpoint definitions" → locates route handlers
- "Configuration management" → finds config files and usage

💡 **Pro tip**: Use this before making any changes to understand existing patterns and conventions.""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Name of the LEANN index to search. Use 'leann_list' first to see available indexes.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search query - can be natural language (e.g., 'how to handle errors') or technical terms (e.g., 'async function definition')",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 5,
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "Number of search results to return. Use 5-10 for focused results, 15-20 for comprehensive exploration.",
                                },
                                "complexity": {
                                    "type": "integer",
                                    "default": 32,
                                    "minimum": 16,
                                    "maximum": 128,
                                    "description": "Search complexity level. Use 16-32 for fast searches (recommended), 64+ for higher precision when needed.",
                                },
                                "show_metadata": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Include file paths and metadata in search results. Useful for understanding which files contain the results.",
                                },
                                "metadata_filters": {
                                    "type": "object",
                                    "description": "Optional LEANN metadata filters, e.g. {'source_type': {'==': 'slack'}}.",
                                },
                                "vector_weight": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 1.0,
                                    "description": "Hybrid weight: 1.0 pure vector, 0.0 pure BM25.",
                                },
                                "prefilter": {
                                    "type": "string",
                                    "enum": ["auto", "always", "never"],
                                    "default": "auto",
                                    "description": "Metadata prefilter routing mode.",
                                },
                                "prefilter_threshold": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 0.05,
                                    "description": "Selectivity threshold for prefilter='auto'.",
                                },
                                "explain_filters": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Include metadata filter routing diagnostics in JSON output.",
                                },
                                "diversify_by": {
                                    "type": "string",
                                    "description": "Metadata field used to cap results per group.",
                                },
                                "max_per_group": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 20,
                                    "default": 2,
                                    "description": "Maximum results per diversify_by group.",
                                },
                            },
                            "required": ["index_name", "query"],
                        },
                    },
                    {
                        "name": "leann_multi_search",
                        "description": """🔎 Batched multi-query LEANN search with RRF fusion.

Use this for prose/chat/meeting recall: send the user's query plus 4-6 paraphrases in
`extra_queries`. LEANN embeds the queries in one batch when possible, runs each search,
then fuses results with reciprocal-rank fusion. Use `search_mode='prose'` for meetings
and notes, `code` for codebases, `exact` for exact keyword lookup, and `filtered` when
metadata filters should get a larger candidate pool.""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Name of the LEANN index to search.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Primary user query, as written.",
                                },
                                "extra_queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional paraphrases/angles for broad recall. Skip for exact names, dates, or quoted phrases.",
                                },
                                "search_mode": {
                                    "type": "string",
                                    "enum": ["prose", "code", "exact", "filtered"],
                                    "default": "prose",
                                    "description": "Default tuning preset. prose uses vector_weight 0.3; code uses 1.0; exact uses 0.0.",
                                },
                                "vector_weight": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Hybrid weight: 1.0 pure vector, 0.0 pure BM25. Overrides search_mode default.",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 12,
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Number of fused results to return unless limit is set.",
                                },
                                "fetch": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 500,
                                    "description": "Per-query candidate pool before RRF fusion. Raise when filtering by speaker/date.",
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                    "description": "Final fused result count. Defaults to top_k.",
                                },
                                "metadata_filters": {
                                    "type": "object",
                                    "description": "Optional LEANN metadata filters.",
                                },
                                "prefilter": {
                                    "type": "string",
                                    "enum": ["auto", "always", "never"],
                                    "default": "auto",
                                },
                                "diversify_by": {"type": "string"},
                                "max_per_group": {"type": "integer", "minimum": 1, "maximum": 20},
                                "context_window": {"type": "integer", "minimum": 0, "maximum": 10},
                                "complexity": {
                                    "type": "integer",
                                    "default": 64,
                                    "minimum": 16,
                                    "maximum": 256,
                                },
                            },
                            "required": ["index_name", "query"],
                        },
                    },
                    {
                        "name": "leann_list",
                        "description": "📋 Show all your indexed codebases - your personal code library! Use this to see what's available for search.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "get_session",
                        "description": """📄 Open an agent session log (Claude Code / Codex / pi-agent) with context-efficient pagination.

Two modes:
  - **anchor**: pass `event_id` (from a search_sessions chunk's source_id) to land directly
    at that event with ±`context_events` surrounding it. This is the recommended flow.
  - **paginate**: pass `start_line` + `line_count` to browse the file linearly.

Default `format=compact` returns one line per event: `[lineno] role/type @timestamp — preview`.
Switch to `format=raw` to get original JSONL (much larger, use sparingly).

The response header always shows: total_lines, returned range, truncation status. Use
`get_session` again with new start_line to walk the file without dumping it all into context.""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_path": {
                                    "type": "string",
                                    "description": "Absolute path to the session file (from search_sessions `extra.session_path`).",
                                },
                                "session_id": {
                                    "type": "string",
                                    "description": "Session id (filename-based glob across ~/.claude/projects, ~/.codex/sessions, ~/.pi/agent/sessions) — used when session_path is not provided.",
                                },
                                "event_id": {
                                    "type": "string",
                                    "description": "Anchor at this event id (substring match within a line). Combines with `context_events`.",
                                },
                                "context_events": {
                                    "type": "integer",
                                    "default": 10,
                                    "minimum": 0,
                                    "maximum": 200,
                                    "description": "When `event_id` is given, return this many events on each side. Default 10 → 21 events total.",
                                },
                                "start_line": {
                                    "type": "integer",
                                    "default": 0,
                                    "minimum": 0,
                                    "description": "0-indexed first line to return (ignored when `event_id` is set).",
                                },
                                "line_count": {
                                    "type": "integer",
                                    "default": 50,
                                    "minimum": 1,
                                    "maximum": 500,
                                    "description": "Max lines to return when not anchored. Default 50.",
                                },
                                "format": {
                                    "type": "string",
                                    "enum": ["compact", "raw"],
                                    "default": "compact",
                                    "description": "compact = one-line preview per event (default, context-cheap). raw = original JSONL.",
                                },
                            },
                        },
                    },
                    {
                        "name": "search_sessions",
                        "description": """🧠 Search across your local AI coding-agent session history (Claude Code, Codex, pi-agent).

Queries the unified `coding-sessions` index built nightly by tools/build_coding_sessions.py.
Supports:
  - source filter (claude_code / codex / pi_agent)
  - project filter (e.g. cadence-pipeline, studymill)
  - natural-language time windows ("last week", "around new year", "in March")

Returns chunks with session_path + session_id in metadata so you can open the full
session log for context expansion via the Read tool.

Examples:
  - "auth error fix" agent=codex                           → debugging in codex sessions
  - "what did claude do in cadence-pipeline last week"    → narrowed by project + time
  - "around new year"                                      → temporal cut across all agents""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query. May include natural-language time phrases.",
                                },
                                "agent": {
                                    "type": "string",
                                    "enum": ["claude_code", "codex", "pi_agent"],
                                    "description": "Restrict to a single agent's sessions.",
                                },
                                "project": {
                                    "type": "string",
                                    "description": "Restrict to sessions launched in a project (matches project_id, e.g. 'studymill', 'cadence-pipeline').",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 50,
                                    "description": "Max results.",
                                },
                                "index_name": {
                                    "type": "string",
                                    "default": "coding-sessions",
                                    "description": "Override target index (default: coding-sessions).",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                ]
            },
        }

    elif request.get("method") == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        try:
            if tool_name == "leann_search":
                # Validate required parameters
                if not args.get("index_name") or not args.get("query"):
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Error: Both index_name and query are required",
                                }
                            ]
                        },
                    }

                cmd = [
                    *_leann_cmd(),
                    "search",
                    args["index_name"],
                    args["query"],
                    f"--top-k={args.get('top_k', 5)}",
                    f"--complexity={args.get('complexity', 32)}",
                    "--non-interactive",
                    "--json",
                ]
                if args.get("show_metadata", False):
                    cmd.append("--show-metadata")
                filters, err = _parse_json_object(args.get("metadata_filters"), "metadata_filters")
                if err:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {"content": [{"type": "text", "text": err}]},
                    }
                if filters:
                    cmd.append(f"--metadata-filters={json.dumps(filters)}")
                for src, flag, cast in (
                    ("vector_weight", "--vector-weight", float),
                    ("vectorWeight", "--vector-weight", float),
                    ("prefilter", "--prefilter", str),
                    ("prefilter_threshold", "--prefilter-threshold", float),
                    ("prefilterThreshold", "--prefilter-threshold", float),
                    ("diversify_by", "--diversify-by", str),
                    ("diversifyBy", "--diversify-by", str),
                    ("max_per_group", "--max-per-group", int),
                    ("maxPerGroup", "--max-per-group", int),
                ):
                    if src in args:
                        cmd.append(f"{flag}={cast(args[src])}")
                if args.get("explain_filters", args.get("explainFilters", False)):
                    cmd.append("--explain-filters")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=_base_dir,
                )

            elif tool_name == "leann_multi_search":
                payload = _direct_multi_search(args)
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, ensure_ascii=False, indent=2),
                            }
                        ]
                    },
                }

            elif tool_name == "leann_list":
                result = subprocess.run(
                    [*_leann_cmd(), "list"],
                    capture_output=True,
                    text=True,
                    cwd=_base_dir,
                )

            elif tool_name == "get_session":
                from pathlib import Path

                session_path = args.get("session_path")
                session_id = args.get("session_id")
                event_id = args.get("event_id")
                context_events = int(args.get("context_events", 10))
                start_line = int(args.get("start_line", 0))
                line_count = int(args.get("line_count", 50))
                fmt = args.get("format", "compact")
                if not session_path and not session_id:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Error: session_path or session_id is required",
                                }
                            ]
                        },
                    }
                resolved: Path | None = None
                if session_path:
                    p = Path(session_path).expanduser()
                    if p.is_file():
                        resolved = p
                if resolved is None and session_id:
                    search_roots = [
                        Path.home() / ".claude" / "projects",
                        Path.home() / ".codex" / "sessions",
                        Path.home() / ".pi" / "agent" / "sessions",
                    ]
                    bare = session_id.split(":", 1)[0]
                    for root in search_roots:
                        if not root.exists():
                            continue
                        for pattern in (
                            f"**/{bare}.jsonl",
                            f"**/*{bare}*.jsonl",
                            f"**/*{bare}*.json",
                        ):
                            hits = list(root.glob(pattern))
                            if hits:
                                resolved = hits[0]
                                break
                        if resolved is not None:
                            break
                if resolved is None:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Error: session not found (path={session_path!r} id={session_id!r})",
                                }
                            ]
                        },
                    }

                with resolved.open("r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
                total = len(lines)

                if event_id:
                    anchor = next(
                        (i for i, line in enumerate(lines) if event_id in line),
                        None,
                    )
                    if anchor is None:
                        return {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"# session: {resolved}\n# total_lines: {total}\n# event_id {event_id!r} not found in this session",
                                    }
                                ]
                            },
                        }
                    lo = max(0, anchor - context_events)
                    hi = min(total, anchor + context_events + 1)
                    mode_desc = f"anchor=event_id:{event_id} (line {anchor}); ±{context_events} → lines {lo}-{hi - 1}"
                else:
                    lo = max(0, min(start_line, total))
                    hi = min(total, lo + max(1, line_count))
                    mode_desc = f"paginate start_line={lo} line_count={hi - lo}"

                selected = lines[lo:hi]

                if fmt == "raw":
                    body = "".join(selected)
                else:
                    rendered: list[str] = []
                    for offset, line in enumerate(selected):
                        line_no = lo + offset
                        try:
                            ev = json.loads(line)
                        except Exception:
                            rendered.append(f"[{line_no}] <unparsed> {line.rstrip()[:200]}")
                            continue
                        ts = (
                            ev.get("timestamp")
                            or ev.get("created_at")
                            or (ev.get("payload") or {}).get("timestamp")
                            or ""
                        )
                        kind = ev.get("type") or "?"
                        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
                        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
                        role = (
                            ev.get("role")
                            or msg.get("role")
                            or payload.get("role")
                            or payload.get("type")
                            or kind
                        )
                        preview_src = (
                            ev.get("content")
                            or payload.get("output")
                            or payload.get("text")
                            or msg.get("content")
                            or ev.get("text")
                            or ""
                        )
                        if isinstance(preview_src, list):
                            parts: list[str] = []
                            for block in preview_src:
                                if isinstance(block, dict):
                                    parts.append(str(block.get("text") or block.get("type") or ""))
                                else:
                                    parts.append(str(block))
                            preview = " ".join(p for p in parts if p)
                        elif isinstance(preview_src, dict):
                            preview = json.dumps(preview_src)[:200]
                        else:
                            preview = str(preview_src)
                        preview = " ".join(preview.split())[:200]
                        ev_id = ev.get("id") or payload.get("id") or payload.get("call_id") or ""
                        rendered.append(
                            f"[{line_no}] {role}/{kind} @{ts} id={ev_id} — {preview}"
                        )
                    body = "\n".join(rendered)

                header = (
                    f"# session: {resolved}\n"
                    f"# total_lines: {total}\n"
                    f"# {mode_desc}\n"
                    f"# format: {fmt}\n"
                    f"# next: get_session(session_path={str(resolved)!r}, start_line={hi}) "
                    f"{'(end of file)' if hi >= total else ''}\n"
                )
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"content": [{"type": "text", "text": header + body}]},
                }

            elif tool_name == "search_sessions":
                if not args.get("query"):
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [{"type": "text", "text": "Error: query is required"}]
                        },
                    }
                index_name = args.get("index_name", "coding-sessions")
                top_k = args.get("top_k", 10)
                filters: dict[str, dict[str, str]] = {}
                if agent := args.get("agent"):
                    filters["source_type"] = {"==": agent}
                if project := args.get("project"):
                    filters["project_id"] = {"==": project}
                cmd = [
                    *_leann_cmd(),
                    "search",
                    index_name,
                    args["query"],
                    f"--top-k={top_k}",
                    "--non-interactive",
                    "--json",
                    "--show-metadata",
                ]
                if filters:
                    cmd.append(f"--metadata-filters={json.dumps(filters)}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=_base_dir
                )

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error: unknown tool '{tool_name}'",
                            }
                        ]
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result.stdout
                            if result.returncode == 0
                            else f"Error: {result.stderr}",
                        }
                    ]
                },
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -1, "message": str(e)},
            }


def main():
    global _base_dir

    parser = argparse.ArgumentParser(description="LEANN MCP server (stdio)")
    parser.add_argument(
        "--base-dir",
        help="Base directory where LEANN indexes are located. "
        "The leann CLI will run with this as its working directory.",
    )
    cli_args = parser.parse_args()
    _base_dir = cli_args.base_dir

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            if response:
                print(json.dumps(response))
                sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -1, "message": str(e)},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
