# Apple Calendar Source

Indexes macOS Apple Calendar events from `~/Library/Calendars/Calendar Cache`.

The source copies the SQLite cache before reading it and emits canonical
SIGNALS metadata:

- `event_time`: calendar start time in UTC
- `event_time_local`: local start time with offset
- `created_at`: calendar creation timestamp when available, otherwise event time
- `modified_at`: calendar update timestamp when available, otherwise event time
- `indexed_at`: stamped by LEANN during index build

```bash
leann index --source apple-calendar
```

The compatibility alias remains available for one release:

```bash
leann index-calendar
```
