# Ingestion Guides

This guide provides practical recipes for indexing various data sources and optimizing LEANN for different scales.

## Personal Data (macOS)

LEANN features built-in connectors for your local communication data.

### Apple Mail
Index your local email database for semantic retrieval:
```bash
leann index-email --max-items 5000
```
*Note: Requires "Full Disk Access" for your Terminal app in macOS System Settings.*

### Apple Calendar
Make your schedule searchable:
```bash
leann index-calendar
```

### iMessage History
Index your chats grouped by conversation:
```bash
leann index-imessage --max-items 2000
```

## Code Intelligence

### AST-Aware Chunking
For deep analysis of source code (Python, JS, TS, etc.), use the AST-aware parser. It ensures logical blocks like functions and classes stay together.

**For complex logic (e.g., Trading Strategies):**
Use a larger chunk size to keep entire functions in one context:
```bash
leann build my-code --docs ~/src --use-ast-chunking --ast-chunk-size 1000
```

## Large Scale Archives

### Using DiskANN
For vaults exceeding 10GB or containing millions of chunks, the **DiskANN** backend is recommended to save RAM:

```bash
leann build big-vault --docs /path/to/archive --backend-name diskann
```

### Deep Directory Optimization
If your folder structure is very deep (e.g., Nextcloud), increase the metadata headroom:

```bash
leann build my-vault --docs ~/Nextcloud \
  --doc-chunk-size 1024 \
  --doc-chunk-overlap 200
```

## Automating Sync
LEANN provides utility scripts in `scripts/sync_utilities/` to keep your vaults updated. You can run these via cron or manual triggers to refresh your data.
