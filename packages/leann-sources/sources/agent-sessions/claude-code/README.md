# Claude Code Sessions

Indexes on-disk Claude Code session logs from `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl`.

Each user/assistant message becomes one chunk. Thinking blocks, tool uses, and tool results are inlined into the message text with explicit tags so they remain searchable.

## Discovery

- Root: `~/.claude/projects/` (override via `data.default_path` in the manifest).
- Each subdirectory is one project (encoded cwd: `/` replaced by `-`).
- Each `*.jsonl` inside is one session.

## cwd resolution

The encoded directory name is lossy (a path like `cadence-pipeline` and `cadence/pipeline` collapse to the same encoded form). The reader prefers the **first `cwd` value found in the session file's event stream** (Claude Code's `system` events carry it natively). The decoded directory name is used only as a fallback when no event records `cwd`.

## Local user identity

The `author` field for user messages comes from `$LEANN_LOCAL_USER` if set, otherwise the neutral string `"local_user"`. No usernames are read from the host environment.

## Skipped events

- `permission-mode`, `ai-title`, `file-history-snapshot`, `last-prompt`, `system`, `attachment` — control/diagnostic events, not message content.
- `isMeta: true` messages — local command caveats, not user-authored content.

## SIGNALS mapping

| SIGNALS field | Source |
|---|---|
| `created_at` | Per-message `timestamp` (UTC ISO) |
| `modified_at` | Session file mtime (UTC ISO) |
| `event_time` | Per-message `timestamp` (UTC ISO) |
| `author` | `$LEANN_LOCAL_USER` (user role) or `claude_code` (assistant) |
| `activity_type` | `authored` |
| `source_type` | `claude_code` |
| `source_id` | `<session-id>:<event-uuid>` |
| `source_document_id` | `agent:claude_code:session:<session-id>` (portable, no local path) |
| `project_id` | Path-string-derived basename of cwd (last two components when basename is generic like `workspace`, `src`, `lib`, etc.) |
| `parent_ref` | `session:<session-id>` |
| `chunk_seq` | Per-session sequence (auto) |

`extra` (reader-specific, free-form): `cwd`, `session_id`, `session_started_at`, `session_path` (absolute local path), `role`, `uuid`, `parent_uuid`, `is_sidechain`, `model`, `provider`, `agent`.
