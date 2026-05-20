# iMessage

Use this source when the user wants to index or query macOS iMessage history
from `~/Library/Messages/chat.db`.

Prefer:

```bash
leann index --source imessage
```

Use `leann sources validate imessage` first when database access may be blocked
by macOS Full Disk Access. Do not suggest copying `chat.db` unless the live
database is locked or unavailable.
