# LEANN Source Catalog

This catalog is the manifest-driven ingest layer used by `leann-sources`. Each source lives in a category folder with:

- `manifest.yaml` — source contract, SIGNALS mapping, privacy tier, and reader class.
- `reader.py` — thin source-specific reader when the shared base reader is not enough.
- `README.md` — user-facing setup notes.
- `SKILL.md` — agent-facing guidance for when to suggest the source.

Regenerate `registry.yaml` after adding or moving a source:

```bash
.venv/bin/python packages/leann-sources/tools/generate_sources_registry.py
```

## Registered Sources

| Source | Category | Status | Data type | Notes |
|---|---|---|---|---|
| `chrome` | browser | migrated | sqlite | Reads Chrome or Brave history databases through the registry path. |
| `apple-calendar` | calendar | migrated | sqlite | Reads Apple Calendar cache data with event-time semantics. |
| `github` | developer-tools | new | api | Reads issues, pull requests, comments, and commits with `GITHUB_TOKEN`. |
| `apple-mail` | email | migrated | filesystem | Reads `.emlx` messages from Apple Mail. |
| `chatgpt-export` | imports | migrated | export_zip | Reads ChatGPT data export archives. |
| `claude-export` | imports | migrated | export_zip | Reads Claude data export archives. |
| `imessage` | messaging | migrated | sqlite | Reads macOS Messages `chat.db`. |
| `wechat` | messaging | migrated | export_zip | Reads WeChat export directories. |
| `whatsapp` | messaging | new | sqlite | Reads backup-extracted WhatsApp `ChatStorage.sqlite`. |

## Status Terms

- `migrated`: former hardcoded `leann index-*` reader now available through `leann index --source`.
- `new`: source added first in the registry package, without an older hardcoded command.

## Compatibility

The migrated sources keep their old `leann index-*` aliases for one release. New source work should target `leann index --source <name>` and avoid adding more hardcoded CLI commands.
