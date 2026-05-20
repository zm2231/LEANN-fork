# Apple Mail Source

Indexes local Apple Mail `.emlx` messages from `~/Library/Mail`.

The reader discovers nested `Messages` directories, preserves the existing
email parsing behavior, and emits canonical SIGNALS metadata from message
headers where available:

- `created_at` / `event_time`: `Date`
- `modified_at`: `X-Last-Modified`
- `author`, `activity_type`, and `participant_ids`: sender and recipients
- `indexed_at`: stamped by LEANN during index build

```bash
leann index --source apple-mail
```

The compatibility alias remains available for one release:

```bash
leann index-email
```
