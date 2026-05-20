# iMessage Source

Indexes macOS Messages `chat.db` into one LEANN chunk per message.

Default database path:

```bash
~/Library/Messages/chat.db
```

macOS may require Full Disk Access for the terminal or service account running
LEANN. The source maps iMessage Cocoa nanosecond timestamps into UTC ISO
SIGNALS fields:

- `event_time`: message send time
- `created_at`: message send time, marked `created_at_synthesized`
- `modified_at`: `date_edited` when present
- `indexed_at`: stamped by LEANN during index build

Use the registry command:

```bash
leann index --source imessage
```

The legacy alias remains available for this compatibility window:

```bash
leann index-imessage
```
