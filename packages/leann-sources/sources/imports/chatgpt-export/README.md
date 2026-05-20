# ChatGPT Export Source

Indexes ChatGPT export HTML or ZIP files.

```bash
leann index --source chatgpt-export ./chatgpt-index --dry-run
leann index-chatgpt --export-path /path/to/chat.html
```

When the export has no distinct modified timestamp, `modified_at` is copied
from `event_time` and marked as synthesized.
