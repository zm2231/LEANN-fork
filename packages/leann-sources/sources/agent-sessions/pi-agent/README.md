# Pi Agent Sessions

Indexes pi-agent on-disk session logs. Default root is `~/.pi/agent/sessions/`; additional roots are supplied via the `LEANN_PI_AGENT_ROOTS` environment variable (comma- or colon-separated).

## Discovery

- Default root: `data.default_path` (`~/.pi/agent/sessions`) plus `data.roots`.
- Extra roots: `LEANN_PI_AGENT_ROOTS=/path/one,/path/two` adds them at instantiation.
- Each root is walked recursively for `*.jsonl`; resolved paths are deduped across roots so overlapping configurations cannot double-count.

Pi-agent stores sessions across many installs. Anything you want indexed must be either declared in the manifest's `data.roots` (registry-shipped default) or contributed via the env var (local override).

## Format

Each session file:
- First event (within first 10 lines): `{"type":"session", "id", "timestamp", "cwd", ...}` — header.
- Subsequent events: `message`, `custom_message`, `model_change`, `thinking_level_change`.

`model_change` and `thinking_level_change` events are not indexed but their values are carried forward and attached to subsequent message chunks as `extra.provider`, `extra.model_id`, `extra.thinking_level`.

`custom_message` events with `display: false` are skipped (metadata-only steering events). Others are emitted with `[custom_message:<customType>] <content>` text and `extra.custom_type` set.

## Local user identity

The `author` field for user messages comes from `$LEANN_LOCAL_USER` if set, otherwise `"local_user"`.

## SIGNALS mapping

| SIGNALS field | Source |
|---|---|
| `created_at` | Per-message `timestamp` (UTC ISO) |
| `modified_at` | Session file mtime (UTC ISO) |
| `event_time` | Per-message `timestamp` (UTC ISO) |
| `author` | `$LEANN_LOCAL_USER` (user role) or `pi_agent` (assistant/other) |
| `activity_type` | `authored` for user/assistant, `created` otherwise |
| `source_type` | `pi_agent` |
| `source_id` | `<session-id>:<event-type>:<event-id>` |
| `source_document_id` | `agent:pi_agent:session:<session-id>` (portable, no local path) |
| `project_id` | Path-string-derived basename of session header `cwd` |
| `parent_ref` | `session:<session-id>` |

`extra` (reader-specific): `cwd`, `session_id`, `session_started_at`, `session_path`, `role`, `event_type`, `custom_type`, `parent_id`, `provider`, `model_id`, `thinking_level`, `agent`.
