# Codex Sessions

Indexes Codex on-disk session rollouts from `$CODEX_HOME/sessions/` (default `~/.codex/sessions/`).

## Discovery

- Root: `$CODEX_HOME/sessions/` if `CODEX_HOME` is set, else `~/.codex/sessions/`.
- Recursive walk for `rollout-*.jsonl` and `rollout-*.json` (legacy).
- Per-file: parses `session_meta` (within the first 10 lines) for `cwd`, `id`, `timestamp`, `cli_version`, `originator`.

## Indexed payloads

`response_item` events with payload types:

| Payload type | What it captures | Text format |
|---|---|---|
| `message` | User/assistant/developer messages | Concatenated `input_text` / `output_text` / `text` blocks |
| `reasoning` | Model reasoning summaries | `[reasoning] ...` (or `(encrypted)` placeholder) |
| `function_call` | Tool calls | `[function_call:<name>] <args>` |
| `function_call_output` | Tool results | `[function_call_output] <output>` |
| `tool_search_call`, `tool_search_output` | Tool search | tagged |

Skipped: `event_msg`, `turn_context`, `session_meta` (parsed for header only).

Tool outputs > 4000 chars are truncated with `extra.truncated=true` and `extra.original_length` set; full content can be retrieved from `extra.session_path`.

## Local user identity

The `author` field for user messages comes from `$LEANN_LOCAL_USER` if set, otherwise `"local_user"`.

## SIGNALS mapping

| SIGNALS field | Source |
|---|---|
| `created_at` | Per-event `timestamp` (UTC ISO) |
| `modified_at` | Rollout file mtime (UTC ISO) |
| `event_time` | Per-event `timestamp` (UTC ISO) |
| `author` | `$LEANN_LOCAL_USER` (user role) or `codex` (everything else) |
| `activity_type` | `authored` for message/reasoning, `created` for tool calls/outputs |
| `source_type` | `codex` |
| `source_id` | `<session-id>:<payload-type>:<event-id>` (payload-type included so `function_call` and `function_call_output` never collide on `call_id`) |
| `source_document_id` | `agent:codex:session:<session-id>` (portable, no local path) |
| `project_id` | Path-string-derived basename of `session_meta.payload.cwd` |
| `parent_ref` | `session:<session-id>` |

`extra` (reader-specific): `cwd`, `session_id`, `session_started_at`, `session_path`, `role`, `payload_type`, `truncated`, `original_length`, `cli_version`, `originator`, `provider`, `agent`.
