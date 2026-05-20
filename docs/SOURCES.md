# LEANN Sources

LEANN sources are declarative ingest adapters. A source is added by writing a manifest plus a thin reader, then registering it in `packages/leann-sources/sources/registry.yaml`.

The current catalog has 9 sources:

| Source | What It Indexes | Setup |
|---|---|---|
| `apple-calendar` | Apple Calendar events from the local calendar cache | Requires local Calendar cache access. |
| `apple-mail` | Apple Mail `.emlx` messages | May require Full Disk Access. |
| `chatgpt-export` | ChatGPT data export archives | Point the source at the export zip. |
| `chrome` | Chrome or Brave browser history | Close the browser if the History database is locked. |
| `claude-export` | Claude data export archives | Point the source at the export zip. |
| `github` | GitHub issues, pull requests, comments, and commits | Set `GITHUB_TOKEN` and `GITHUB_REPOSITORIES`. |
| `imessage` | macOS Messages from `~/Library/Messages/chat.db` | May require Full Disk Access. |
| `wechat` | WeChat export directories | Point the source at the export directory. |
| `whatsapp` | WhatsApp `ChatStorage.sqlite` from an iOS backup | Extract the database from a local backup first. |

## Commands

List registered sources:

```bash
leann sources list
```

Inspect one source:

```bash
leann sources info github
```

Validate local data or auth before indexing:

```bash
leann sources validate github
```

Index through the unified entry point:

```bash
leann index --source github my-github-index
```

## Metadata Contract

Readers emit SIGNALS-compliant chunks. Time-bearing sources should populate `event_time`; source-created and source-edited timestamps should populate `created_at` and `modified_at` when the source distinguishes them. If a source copies one axis from another, it should also emit `<axis>_synthesized: true`.

Source identity fields are consistent across the catalog:

- `source_type` identifies the source family, such as `github`, `whatsapp`, or `browser_history`.
- `source_id` is stable within the source.
- `source_document_id` groups chunks for `diversify_by` and `context_window`.
- `source_url`, `parent_ref`, `mentioned_urls`, `mentioned_refs`, and `mentioned_files` provide connector-ready references.

## Adding A Source

Add a new source under `packages/leann-sources/sources/<category>/<name>/`:

1. Write `manifest.yaml` with data, auth, fields, chunking, privacy, and connector hints.
2. Add `reader.py` only when the shared base reader is not enough.
3. Add `README.md` and `SKILL.md`.
4. Add fixture tests for `validate()` and `iter_chunks()`.
5. Regenerate `registry.yaml`.

Do not add another hardcoded `leann index-*` command. New ingest surfaces should use `leann index --source <name>`.
