#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys

_base_dir: str | None = None


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
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=_base_dir,
                )

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
                    cmd.append(f"--metadata-filter={json.dumps(filters)}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=_base_dir
                )

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
