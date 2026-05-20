# Claude Export Source

Indexes Claude export JSON or ZIP files.

```bash
leann index --source claude-export
leann index-claude --export-path /path/to/claude.json
```

When the export has no distinct modified timestamp, `modified_at` is copied
from `event_time` and marked as synthesized.
