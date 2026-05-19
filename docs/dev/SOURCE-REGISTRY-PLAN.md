# SOURCE-REGISTRY-PLAN — Wave 3

**Status:** drafted, not started. Branch name when started: `feat/source-registry`. Off `zain/custom-patches` (or `origin/main` + cherry-pick — decide at dispatch time).

**Goal:** turn LEANN's hardcoded `index-*` commands into a **declarative source registry** so any new data source (WhatsApp, GitHub, Notion, Linear, voice memos, etc.) can be added by writing a manifest + thin reader, not a one-off CLI command. Modeled on Printing Press Library's catalog pattern (manifest-driven library, auto-generated registry, per-entry SKILL.md, install command), adapted for *ingest* surfaces instead of action surfaces.

**Why:** Wave 1 + Wave 2 gave us the SIGNALS schema + filter/search primitives. We have 7 hardcoded readers (`index-email`, `index-imessage`, `index-calendar`, `index-browser`, `index-chatgpt`, `index-claude`, `index-wechat`) — each duplicates discovery + auth + chunking + SIGNALS-emission logic. Adding an 8th source is a copy-paste job today. After Wave 3, it's a manifest edit + ~50 LOC reader.

**Inspiration:** `github.com/mvanhorn/printing-press-library` — 134 CLIs across 17 categories, each declared by `manifest.json` + `SKILL.md`, indexed by auto-generated `registry.json`. PP catalogs *action* APIs; LEANN sources are *ingest* APIs. Same catalog shape, different reader interface.

---

## Glossary

- **Source** — any external system LEANN ingests from (iMessage, Slack, WhatsApp, GitHub, email, browser, calendar, voice memos). Each has its own data shape and access pattern.
- **Source manifest** — `apps/sources/<category>/<name>/manifest.yaml`. Declarative contract: where the raw data lives, how to authenticate, what fields it exposes, how those fields map to SIGNALS, importance signals, chunking strategy, privacy tier.
- **`SourceReader`** — Python protocol every source implements. Methods: `discover()`, `validate()`, `iter_chunks()`, `stats()`. Subclass shortcuts: `SQLiteSourceReader`, `FilesystemSourceReader`, `APISourceReader`, `ExportZipSourceReader`.
- **Source registry** — `apps/sources/registry.json`, auto-generated from all manifests. Top-level catalog the CLI reads.
- **Field mapping** — the manifest section that says "the source's `messages.ts` column becomes `event_time` via `unix_to_utc_iso` transform".
- **Importance signals** — manifest section that declares what carries weight in this source: recency decay rate, participant-count weighting, flagged keyword list, thread depth multiplier. Connector layer (Wave 6) consumes these.
- **Privacy tier** — `tier_1` (user-only), `tier_2` (metadata-only allowed for sharing), `tier_3` (multi-party content, content excluded from shared profile). From context-layer naming.
- **`leann sources` CLI namespace** — new top-level command group: `list`, `install`, `info`, `connect`, `validate`. Plus unified `leann index --source <name>`.

---

## Architecture overview

```
apps/sources/
  registry.json                        ← auto-generated index of all sources
  messaging/
    imessage/                          ← migrated from leann index-imessage
      manifest.yaml
      reader.py                        ← subclasses SQLiteSourceReader
      SKILL.md                         ← Claude Code skill
      README.md
      tests/
        fixtures/sample_chat.db
        test_reader.py
    whatsapp/                          ← NEW
      ...
    slack/                             ← NEW (replaces .slacrawl ingest workaround)
      ...
  email/
    apple-mail/                        ← migrated from leann index-email
      ...
  calendar/
    apple-calendar/                    ← migrated from leann index-calendar
      ...
  browser/
    chrome/                            ← migrated from leann index-browser
      ...
  code/
    github/                            ← NEW
      ...
    local-git-log/                     ← extracted from Wave 1 eval corpus builder
      ...
  productivity/
    notion/                            ← NEW
      ...
    linear/                            ← NEW
      ...
  imports/
    chatgpt-export/                    ← migrated from leann index-chatgpt
      ...
    claude-export/                     ← migrated from leann index-claude
      ...
  voice/
    voice-memos/                       ← NEW (pairs with Whisper)
      ...

packages/leann-core/src/leann/sources/
  __init__.py
  base.py                              ← SourceReader protocol + base classes
  manifest.py                          ← manifest schema + loader
  registry.py                          ← discovery + generate_registry()
  transforms.py                        ← canonical field transforms (unix_to_utc_iso,
                                         core_data_epoch_to_utc_iso, regex_extract, etc.)
```

---

## Manifest schema (the load-bearing artifact)

`apps/sources/<category>/<name>/manifest.yaml`:

```yaml
name: whatsapp
category: messaging
display_name: "WhatsApp"
version: "0.1.0"
manifest_version: "1.0"

# Where the raw data lives
data:
  type: sqlite                      # sqlite|filesystem|api|export_zip|cloud_drive|live_stream
  default_path: "~/Library/Application Support/WhatsApp/ChatStorage.sqlite"
  permissions:                      # macOS TCC etc. surfaced by `leann sources validate`
    - "Full Disk Access"

# How the user authenticates (if applicable)
auth:
  type: none                        # none|env_var|keychain|oauth|api_key
  # for env_var/api_key:
  # env_vars: [WHATSAPP_TOKEN]
  # for oauth:
  # provider: github
  # scopes: [repo, read:user]

# What each chunk exposes — the SIGNALS mapping
fields:
  event_time:
    source: "ZMESSAGE.ZMESSAGEDATE"
    transform: "core_data_epoch_to_utc_iso"
    required: true
  author:
    source: "ZMESSAGE.ZFROMHANDLE"
    transform: "lookup(ZHANDLE.ZID)"
  source_type:
    value: "whatsapp"               # constant — no source column needed
  source_id:
    source: "ZMESSAGE.Z_PK"
  parent_ref:
    source: "ZCHATSESSION.ZGROUPNAME"
    transform: "format('chat:{value}')"
  participant_ids:
    source: "ZCHATSESSION.participants"
    transform: "list_of_handles"
  mentioned_urls:
    extraction: "regex"
    pattern: '<https?://[^\\s>]+'
  mentioned_refs:
    extraction: "regex"
    pattern: '@\\w+'

# Importance signals — what this source carries that scoring/connector layers use
importance:
  recency_decay: "default"            # default|fast|slow|none
  participant_count_weight: 0.8       # 1:1 chat > broadcast
  thread_depth_weight: 0.5
  flagged_keywords: ["urgent", "ASAP", "review"]

# Chunking
chunking:
  granularity: "message"              # message|thread|day|conversation|file
  thread_grouping_window_sec: 600     # ≤10min = same thread
  max_chunk_tokens: 400
  overlap_tokens: 0

# Privacy / consent
privacy:
  tier: "tier_3"                      # tier_1=mine|tier_2=metadata_only|tier_3=multi_party_content
  participant_consent_required: true
  content_excluded_in: ["profile_shareable"]

# Connector hints (Wave 6+)
connectors:
  cross_source_match_by:
    - mentioned_urls                  # bridges to browser_history
    - participant_ids                 # bridges to email, slack, imessage
  event_id_strategy: "synthetic_thread_hash"
```

Validated by `packages/leann-core/src/leann/sources/manifest.py` at load time.

---

## SourceReader interface

`packages/leann-core/src/leann/sources/base.py`:

```python
class SourceReader(Protocol):
    manifest: SourceManifest

    def discover(self) -> DiscoveryResult: ...
    """Find the raw data on the system. Returns paths/URLs found,
       what was checked, what's missing."""

    def validate(self) -> ValidationReport: ...
    """Check accessibility, permissions, schema compatibility.
       Manifest version vs. observed schema version, TCC permission
       state, auth credentials presence."""

    def iter_chunks(self, since: datetime | None = None) -> Iterator[Chunk]: ...
    """Yield SIGNALS-compliant chunks. Honors `since` for incremental ingest."""

    def stats(self) -> SourceStats: ...
    """Count, date range, top values for filter fields — used by
       `leann sources info`."""
```

Three concrete base classes cover ~80% of cases:
- `SQLiteSourceReader` — connects to a SQLite DB, runs a SELECT, applies field transforms per manifest. Used by imessage, whatsapp, chrome history, voice memos.
- `FilesystemSourceReader` — walks a directory, applies file filters, optionally parses per-extension. Used by apple-mail (.emlx), local-git-log, journals.
- `APISourceReader` — paginated HTTP fetch with auth + rate limiting. Used by github, notion, linear.
- `ExportZipSourceReader` — extracts a zip + iterates JSON/HTML files. Used by chatgpt-export, claude-export.

Each base class consumes the manifest's `data` + `fields` sections automatically — concrete reader subclass only overrides what's unusual.

---

## CLI surface

```bash
# Discovery
leann sources list                          # all sources, status per source
leann sources list --category messaging     # filter
leann sources info whatsapp                 # manifest + sample chunk + stats

# Setup
leann sources install whatsapp              # writes config, validates deps
leann sources connect whatsapp              # auth flow if needed (oauth, keychain)
leann sources validate whatsapp             # dry-run: can we read?

# Ingest
leann index --source whatsapp my-whatsapp                  # unified entry point
leann index --source whatsapp my-whatsapp --since 7d       # incremental
leann index --source whatsapp my-whatsapp --dry-run        # iterate without indexing

# Backwards compat (one release, then deprecate)
leann index-email          # → alias for `leann index --source apple-mail`
leann index-imessage       # → alias for `leann index --source imessage`
# ...etc
```

The unified `leann index --source` accepts ALL existing `leann build` embedding/chunking flags, applied to whatever the source emits.

---

## Atoms (sequenced)

### Atom 0 — Schema + base classes (no source migrations yet)

**What:**
- `packages/leann-core/src/leann/sources/__init__.py`
- `packages/leann-core/src/leann/sources/manifest.py` — `SourceManifest` dataclass + YAML loader + validator (jsonschema-style)
- `packages/leann-core/src/leann/sources/base.py` — `SourceReader` protocol + `Chunk`, `DiscoveryResult`, `ValidationReport`, `SourceStats` dataclasses
- `packages/leann-core/src/leann/sources/transforms.py` — canonical transforms: `unix_to_utc_iso`, `core_data_epoch_to_utc_iso`, `webkit_epoch_to_utc_iso`, `regex_extract`, `format`, `lookup`, `list_of_handles`
- Manifest schema spec at `apps/sources/SCHEMA.md`

**Done test:** `tests/test_sources_manifest.py` — round-trip load + validate a sample manifest with all field types + invalid cases.

**Commit:** `feat(sources): manifest schema + SourceReader protocol + canonical transforms`

---

### Atom 1 — Base reader classes

**What:** `packages/leann-core/src/leann/sources/readers/`:
- `sqlite.py` — `SQLiteSourceReader` consumes manifest, applies SELECT + transforms
- `filesystem.py` — `FilesystemSourceReader`
- `api.py` — `APISourceReader` (with pagination + rate limit hooks)
- `export_zip.py` — `ExportZipSourceReader`

Each handles 80% of its category via manifest declaration; subclasses override the 20%.

**Done test:** `tests/test_sources_base_readers.py` — each base class with a synthetic manifest + fixture data emits SIGNALS-compliant chunks.

**Commit:** `feat(sources): sqlite/filesystem/api/export_zip base readers`

---

### Atom 2 — Registry + discovery

**What:**
- `packages/leann-core/src/leann/sources/registry.py` — walks `apps/sources/` to find manifests, validates them, builds an in-memory registry, optionally writes `apps/sources/registry.json`
- `tools/generate_sources_registry.py` — invoked by pre-commit or CI to regenerate the JSON index (same pattern as PP's `tools/generate-registry`)

**Done test:** `tests/test_sources_registry.py` — synthetic source dir with 3 manifests; registry discovery returns all 3 with correct categories and validates schema.

**Commit:** `feat(sources): registry discovery + auto-generated catalog`

---

### Atom 3 — CLI namespace

**What:** add `leann sources` subparser group to `packages/leann-core/src/leann/cli.py`:
- `sources list [--category C]`
- `sources info <name>` — manifest + stats
- `sources install <name>` — validate deps, create config dir
- `sources connect <name>` — auth flow (calls reader's `auth_setup()` if defined)
- `sources validate <name>` — dry-run discover + validate

Plus `leann index --source <name>` as a unified entry point.

**Done test:** `tests/test_cli_sources.py` — each subcommand against synthetic registry.

**Commit:** `feat(cli): leann sources namespace + unified index --source`

---

### Atom 4 — Migrate `index-imessage` as proof

**What:** `apps/sources/messaging/imessage/`:
- `manifest.yaml` — full field mapping for chat.db
- `reader.py` — `class IMessageReader(SQLiteSourceReader)`, minimal overrides
- `SKILL.md`
- `README.md`
- `tests/test_imessage_reader.py` with a tiny fixture chat.db

Make `leann index-imessage` an alias for `leann index --source imessage` (one release of compat).

**Done test:** existing iMessage test cases still pass through the new path; new reader emits same SIGNALS schema as old `index-imessage`.

**Commit:** `feat(sources): migrate iMessage to source registry`

---

### Atom 5 — Migrate remaining 6 hardcoded readers

**What:** same pattern as atom 4 for: `apple-mail`, `apple-calendar`, `chrome` (browser), `chatgpt-export`, `claude-export`, `wechat`. One commit per migration.

**Done test:** all 7 old `index-*` commands work via the new registry; backwards-compat aliases stay green.

**Commits:** `feat(sources): migrate <name> to source registry` × 6

---

### Atom 6 — Add WhatsApp + GitHub as first NEW sources

**What:**
- `apps/sources/messaging/whatsapp/` — sqlite reader, may need backup-extraction docs
- `apps/sources/code/github/` — API reader (issues, PRs, comments, commits) with PAT auth

Done test: each can `validate` + `iter_chunks` against a fixture; integration test against real data is a manual step.

**Commits:** 
- `feat(sources): WhatsApp via local ChatStorage.sqlite`
- `feat(sources): GitHub via REST API`

---

### Atom 7 — Skills + docs

**What:** per-source `SKILL.md` files so Claude Code knows when to suggest each. Top-level `apps/sources/README.md` listing all sources with status. `docs/SOURCES.md` aimed at humans.

**Commit:** `docs(sources): per-source skills + catalog README`

---

### Atom 8 — Eval + close

**What:** validate that:
- Registry discovers all migrated sources
- Each emits SIGNALS-compliant chunks via Wave 1's `test_signals_schema.py` (extended to iterate over all manifests)
- Existing Wave 1 + Wave 2 narrow regressions still green
- Per-source `validate` reports human-readable status (works / missing data / missing auth / wrong permissions)

Done bar: 9 sources registered (7 migrated + 2 new), all pass schema validation, all 99 Wave 1 + 2 narrow tests still green.

**Commit:** `docs(dev): wave 3 closed — source registry`

---

## /loop driver prompt

When ready to dispatch, paste this:

```
/loop

Execute Wave 3 of docs/dev/SOURCE-REGISTRY-PLAN.md on branch
feat/source-registry. Atoms 0-8 in strict order.

Required reading before each atom:
- docs/dev/SOURCE-REGISTRY-PLAN.md (the atom)
- docs/dev/SIGNALS.md (chunks MUST be SIGNALS-compliant)
- docs/UPGRADE.md (current API surface — don't break it)

Setup invariants — halt if any fail:
- cwd = (wherever the worktree is)
- git branch = feat/source-registry
- .venv/bin/python -c "import leann, dateparser, yaml" succeeds
- Wave 1+2 narrow regression passes:
  .venv/bin/pytest tests/test_metadata_filter_datetime.py \
                   tests/test_indexed_at.py tests/test_signals_schema.py \
                   tests/test_temporal_parser.py tests/test_search_temporal.py \
                   tests/test_react_temporal.py tests/test_facets.py \
                   tests/test_score_filtered_subset.py \
                   tests/test_prefilter_integration.py \
                   tests/test_explain_filters.py tests/test_diversify.py \
                   tests/test_context_window.py -x
- pkill -f hnsw_embedding_server 2>/dev/null || true

Hard NO list:
- Breaking the existing leann build / index-* CLI surface (atom 4-5 must
  keep them as backwards-compat aliases)
- Renaming any existing public API (LeannBuilder, LeannSearcher, ReActAgent)
- Touching backends (packages/leann-backend-*)
- Touching Wave 1 temporal code or Wave 2 metadata-aware code
- Adding non-SIGNALS metadata fields without updating SIGNALS.md first
- pip installing new deps without asking

Per-atom workflow:
1. Read the atom + relevant SIGNALS.md sections.
2. State in one sentence what you'll change.
3. Edit + write test file.
4. .venv/bin/pytest <new test> -v
5. Run the Wave 1+2 narrow regression (above) — must stay green.
6. .venv/bin/ruff check --fix && .venv/bin/ruff format
7. git add <files>; git commit -m "<documented msg>"
8. Append to docs/dev/PROGRESS-wave3.md: "Atom N: <sha> — <one-line result>"

Stop and ask when:
- An atom's done-test fails 3 times
- Existing Wave 1+2 test regresses (any of the 99)
- Edit needs to touch a file NOT in the atom's "Where" section
- Manifest schema decisions surface (YAML vs JSON, versioning strategy)

Tools: .venv/bin/{python,pytest,ruff}.
```

---

## Open questions (decide before atom 0)

1. **Manifest format:** YAML (more readable for field mapping) vs JSON (no extra dep, more validatable). **Default: YAML** with `pyyaml` as a leann-core dep.
2. **Versioning:** every manifest has `manifest_version` (schema version) AND `version` (source-specific). Schema validation rejects unknown major versions. **Default: yes, do both.**
3. **Scope split:** all 9 atoms one wave, or 3a (atoms 0-3, registry + base classes) + 3b (4-5, migrations) + 3c (6-8, new sources + close)? **Default: one wave, but with the option to halt cleanly between atom 3 and atom 4 if scope feels large.**
4. **Importance signals consumption:** Wave 3 declares them but doesn't *use* them (consumption is Wave 6 connector layer / leann-memory). **Default: declare-only for now; readers don't filter or weight by importance yet.**
5. **PP-style installer:** do we want `leann sources install` to also handle CLI/Skill installation (like PP's npm orchestrator)? **Default: no — leann-core ships everything. The `install` command just validates deps and creates config.**

---

## Out of scope — for waves after this

- **Importance-aware retrieval scoring** — uses Wave 3's importance signals to weight results. Belongs in Wave 5/6 alongside temporal decay.
- **Cross-source connector layer** — uses Wave 3's `connectors.cross_source_match_by` declarations to materialize edges. Wave 6.
- **OAuth flow polish** — atom 3's `sources connect` covers the basic case; a proper credentials manager (Keychain integration, refresh-token handling) is its own mini-wave later.
- **Source-specific MCP servers** — Printing Press wraps each catalog entry as both CLI + MCP. We're not doing MCP per source initially; LEANN's existing single MCP server is enough.
- **Live-stream / push sources** — Wave 3 is pull-based. Push-based sources (webhooks, Kafka, Slack events) come later if needed.
- **Sharing/publishing manifests** — PP has `printingpress.dev`; we can publish ours later if it's worth it.
