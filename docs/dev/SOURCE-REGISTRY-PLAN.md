# SOURCE-REGISTRY-PLAN — Wave 3

**Status:** ready to dispatch. Branch when started: `feat/source-registry` (off `feat/multi-axis-temporal`). Wave 1 + 1.5 + 2 must be in place — they are (HEAD `8199817`).

**Goal:** turn LEANN's hardcoded `index-*` commands into a **declarative source registry** so any new data source (WhatsApp, GitHub, Notion, Linear, voice memos, etc.) can be added by writing a manifest + thin reader, not a one-off CLI command. Modeled on Printing Press Library's catalog pattern (manifest-driven library, auto-generated registry, per-entry SKILL.md, install command), adapted for *ingest* surfaces instead of action surfaces.

**Why:** Wave 1 + Wave 1.5 + Wave 2 gave us the SIGNALS schema (four temporal axes + identity + referential signals) and filter/search primitives. We have 7 hardcoded readers (`index-email`, `index-imessage`, `index-calendar`, `index-browser`, `index-chatgpt`, `index-claude`, `index-wechat`) — each duplicates discovery + auth + chunking + SIGNALS-emission logic. Adding an 8th source is a copy-paste job today. After Wave 3, it's a manifest edit + ~50 LOC reader.

**Package shape:** new package `packages/leann-sources/` (parallel to `leann-backend-hnsw`, `leann-backend-diskann`, `leann-backend-ivf`). Opt-in install. Carries its own deps (`pyyaml`, `jsonschema`). Discovered by `leann-core` via entry points. **Not vendored into `leann-core`.**

**Inspiration:** `github.com/mvanhorn/printing-press-library` — 134 CLIs across 17 categories, each declared by `manifest.json` + `SKILL.md`, indexed by auto-generated `registry.json`. PP catalogs *action* APIs; LEANN sources are *ingest* APIs. Same catalog shape, different reader interface. We adopt PP's directory layout, registry generation, per-entry SKILL.md convention. We don't adopt PP's Go-binary-per-entry or npm orchestration — LEANN readers are Python classes.

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

Everything lives in the new package. `apps/` stays untouched (it's example RAG scripts).

```
packages/leann-sources/                ← NEW PACKAGE
  pyproject.toml                       ← own deps: pyyaml, jsonschema, requests
  README.md
  src/leann_sources/
    __init__.py
    base.py                            ← SourceReader protocol + dataclasses
    manifest.py                        ← SourceManifest + YAML loader + jsonschema validator
    registry.py                        ← discovery + generate_registry()
    transforms.py                      ← canonical field transforms (unix_to_utc_iso,
                                         core_data_epoch_to_utc_iso, regex_extract, lookup, ...)
    cli.py                             ← `leann sources ...` subparser group
    readers/
      __init__.py
      sqlite.py                        ← SQLiteSourceReader
      filesystem.py                    ← FilesystemSourceReader
      api.py                           ← APISourceReader (pagination + rate-limit hooks)
      export_zip.py                    ← ExportZipSourceReader

  sources/                             ← THE CATALOG (PP-style library/)
    registry.yaml                      ← auto-generated index of all sources
    SCHEMA.md                          ← manifest spec
    messaging/
      imessage/                        ← migrated from leann index-imessage
        manifest.yaml
        reader.py                      ← subclasses SQLiteSourceReader
        SKILL.md                       ← Claude Code skill
        README.md
        tests/
          fixtures/sample_chat.db
          test_reader.py
      whatsapp/                        ← NEW
      slack/                           ← NEW (replaces .slacrawl ingest workaround)
      wechat/                          ← migrated from leann index-wechat
    email/
      apple-mail/                      ← migrated from leann index-email
    calendar/
      apple-calendar/                  ← migrated from leann index-calendar
      gcal/                            ← deferred — Wave 3.1 candidate
    browser/
      chrome/                          ← migrated from leann index-browser
    developer-tools/
      github/                          ← NEW
      local-git-log/                   ← extracted from Wave 1 eval corpus builder
    productivity/
      notion/                          ← NEW (likely; pending user pick)
      linear/                          ← NEW (likely; pending user pick)
    imports/
      chatgpt-export/                  ← migrated from leann index-chatgpt
      claude-export/                   ← migrated from leann index-claude
    voice/
      voice-memos/                     ← NEW (pairs with Whisper, post-Wave-3)

packages/leann-core/src/leann/
  cli.py                               ← old `index-<source>` commands become thin aliases that
                                         dispatch to `leann sources index <name>` IF leann-sources
                                         is installed; otherwise error with install hint.
```

**Entry-point registration:** `leann-sources` registers itself with `leann-core` via `entry_points = {"leann.plugins": ["sources = leann_sources:get_plugin"]}`. Same pattern as the backend packages. `leann-core` checks for the plugin at CLI startup; if present, `leann sources ...` and `leann index --source ...` are wired up; if not, the old `index-<source>` commands print an "install leann-sources" message.

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

# What each chunk exposes — the SIGNALS mapping (Wave 1.5 four-axis temporal)
fields:
  # Temporal axes — populate as many as the source actually distinguishes
  event_time:
    source: "ZMESSAGE.ZMESSAGEDATE"
    transform: "core_data_epoch_to_utc_iso"
    required: true
  created_at:
    source: "ZMESSAGE.ZMESSAGEDATE"        # for messages, == event_time
    transform: "core_data_epoch_to_utc_iso"
    synthesized_from: "event_time"         # marks axis as degenerate (Wave 1.5)
  modified_at:
    source: "ZMESSAGE.ZEDITDATE"           # populated only if message was edited
    transform: "core_data_epoch_to_utc_iso"
    required: false
  # indexed_at is auto-stamped by the builder — do not declare here

  # Identity
  author:
    source: "ZMESSAGE.ZFROMHANDLE"
    transform: "lookup(ZHANDLE.ZID)"
  participant_ids:
    source: "ZCHATSESSION.participants"
    transform: "list_of_handles"
  source_type:
    value: "whatsapp"                      # constant — no source column needed
  source_id:
    source: "ZMESSAGE.Z_PK"

  # Document hierarchy (Wave 2)
  source_document_id:
    source: "ZCHATSESSION.Z_PK"            # the chat thread is the parent document
    transform: "format('whatsapp_chat_{value}')"
  chunk_seq:
    auto: "row_index_within_source_document_id"

  # Referential signals
  parent_ref:
    source: "ZCHATSESSION.ZGROUPNAME"
    transform: "format('chat:{value}')"
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

### Atom 0 — New package skeleton

**What:**
- `packages/leann-sources/pyproject.toml` — package name `leann-sources`, deps `pyyaml`, `jsonschema`, `requests`. Entry point: `leann.plugins = sources = leann_sources:get_plugin`.
- `packages/leann-sources/README.md` — what this package is, install via `uv pip install -e packages/leann-sources/`.
- `packages/leann-sources/src/leann_sources/__init__.py` — exports `get_plugin()`, `SourceManifest`, `SourceReader`, `Chunk`.
- `packages/leann-sources/src/leann_sources/manifest.py` — `SourceManifest` dataclass + YAML loader + jsonschema validator.
- `packages/leann-sources/src/leann_sources/base.py` — `SourceReader` protocol + `Chunk`, `DiscoveryResult`, `ValidationReport`, `SourceStats` dataclasses.
- `packages/leann-sources/src/leann_sources/transforms.py` — canonical transforms: `unix_to_utc_iso`, `core_data_epoch_to_utc_iso`, `webkit_epoch_to_utc_iso`, `regex_extract`, `format`, `lookup`, `list_of_handles`. Each maps Wave 1.5 axes correctly (UTC ISO 8601 with offset).
- `packages/leann-sources/sources/SCHEMA.md` — manifest spec, documents the four-axis temporal model.

**Done test:** `packages/leann-sources/tests/test_manifest.py` — round-trip load + validate a sample manifest with all field types + invalid cases (missing required fields, unknown axes, malformed transforms).

**Commit:** `feat(sources): leann-sources package skeleton + manifest schema + SourceReader protocol`

---

### Atom 1 — Base reader classes

**What:** `packages/leann-sources/src/leann_sources/readers/`:
- `sqlite.py` — `SQLiteSourceReader` consumes manifest, applies SELECT + transforms
- `filesystem.py` — `FilesystemSourceReader` (stamps `created_at` from `st_birthtime`, `modified_at` from `st_mtime`)
- `api.py` — `APISourceReader` (with pagination + rate limit hooks)
- `export_zip.py` — `ExportZipSourceReader`

Each handles 80% of its category via manifest declaration; subclasses override the 20%. All readers honor Wave 1.5 axes — if the manifest declares `created_at` + `modified_at` + `event_time`, all three get emitted with `<axis>_synthesized` markers where the source can't distinguish.

**Done test:** `packages/leann-sources/tests/test_base_readers.py` — each base class with a synthetic manifest + fixture data emits SIGNALS-compliant chunks. Schema regression confirms all four temporal axes parse and Wave 1.5 fallback chain works against the output.

**Commit:** `feat(sources): sqlite/filesystem/api/export_zip base readers`

---

### Atom 2 — Registry + discovery + entry-point wiring

**What:**
- `packages/leann-sources/src/leann_sources/registry.py` — walks `packages/leann-sources/sources/` to find manifests, validates them, builds in-memory registry, regenerates `registry.yaml`.
- `packages/leann-sources/src/leann_sources/__init__.py::get_plugin()` — entry-point hook. Returns a `Plugin` object with `cli_subcommands`, `index_source_handler`, etc., so `leann-core`'s CLI can discover and route.
- `packages/leann-sources/tools/generate_sources_registry.py` — pre-commit/CI hook to regenerate `registry.yaml`. Pattern stolen from PP's `tools/generate-registry`.
- `packages/leann-core/src/leann/cli.py` — patch to detect the `leann.plugins` entry-point group at startup and wire registered plugins. Without `leann-sources` installed, behavior unchanged.

**Done test:** `packages/leann-sources/tests/test_registry.py` — synthetic source dir with 3 manifests; registry discovery returns all 3 with correct categories and validates schema. Entry-point discovery test in `packages/leann-core/tests/test_plugin_loading.py` confirms `leann-core` finds `leann-sources` when installed and ignores it cleanly when absent.

**Commit:** `feat(sources): registry discovery + entry-point plugin wiring`

---

### Atom 3 — CLI namespace

**What:** `leann sources` subparser group lives in `leann-sources` (not core), wired into `leann-core`'s CLI via the plugin protocol from atom 2:
- `sources list [--category C]`
- `sources info <name>` — manifest + stats
- `sources install <name>` — validate deps (Python packages, system perms, env vars), create config dir
- `sources connect <name>` — auth flow (calls reader's `auth_setup()` if defined)
- `sources validate <name>` — dry-run discover + validate
- `sources index <name>` — equivalent to `leann index --source <name>`

`leann index` accepts `--source <name>` as an alternative to `--docs <path>`.

**Done test:** `packages/leann-sources/tests/test_cli.py` — each subcommand against synthetic registry. Integration test: `leann sources list` works only when `leann-sources` is installed; clean error otherwise.

**Commit:** `feat(sources): leann sources CLI namespace + unified index --source`

---

### Atom 4 — Migrate `index-imessage` as proof

**What:** `packages/leann-sources/sources/messaging/imessage/`:
- `manifest.yaml` — full field mapping for chat.db, with all four temporal axes (`event_time` = msg.date, `created_at` = msg.date, `modified_at` = msg.date_edited when present, `created_at_synthesized: true` since == event_time)
- `reader.py` — `class IMessageReader(SQLiteSourceReader)`, minimal overrides
- `SKILL.md`
- `README.md`
- `tests/test_imessage_reader.py` with a tiny fixture chat.db

Old `leann index-imessage` becomes a thin alias that calls `leann index --source imessage`. One release of compat; the alias prints a deprecation note pointing at the new command.

**Done test:** existing iMessage test cases still pass through the new path; new reader emits SIGNALS schema with all four axes; schema regression in `packages/leann-core/tests/test_signals_schema.py` confirms compat.

**Commit:** `feat(sources): migrate iMessage to source registry`

---

### Atom 5 — Migrate remaining 6 hardcoded readers

**What:** same pattern as atom 4 for: `apple-mail` → email/, `apple-calendar` → calendar/, `chrome` → browser/, `chatgpt-export` + `claude-export` → imports/, `wechat` → messaging/. Each migration:
1. Write manifest with all Wave 1.5 axes (mark synthesized where applicable).
2. Subclass the right base reader.
3. Port SKILL.md + README.md.
4. Add reader-specific tests.
5. Mark old `leann index-<name>` as deprecated alias.

One commit per migration. The old reader code in `apps/*.py` and `packages/leann-core/src/leann/cli.py::index_*` stays as the alias backend for now (deleted in Wave 3 close-out atom 8 once all migrations are green).

**Done test:** all 7 old `index-*` commands work via the new registry; backwards-compat aliases stay green; SIGNALS regression covers all migrated sources.

**Commits:** `feat(sources): migrate <name> to source registry` × 6

---

### Atom 6 — Add WhatsApp + GitHub as first NEW sources

**What:**
- `packages/leann-sources/sources/messaging/whatsapp/` — sqlite reader against `ChatStorage.sqlite`, with backup-extraction docs in README.md (since iOS doesn't expose it directly). Manifest declares all 4 axes; `modified_at` populated from edit history when iOS 16+.
- `packages/leann-sources/sources/developer-tools/github/` — APISourceReader against REST API. Issues, PRs, comments, commits. PAT auth via `GITHUB_TOKEN` env var. Manifest maps GitHub timestamps: `created_at` = issue.created_at, `modified_at` = issue.updated_at, `event_time` = created_at (events are atomic). Rate-limit hooks via `X-RateLimit-Remaining` header.

**Done test:** each source `validate`s + `iter_chunks` against a fixture; integration test against real data is a manual step (user runs `leann sources connect github && leann index --source github`).

**Commits:** 
- `feat(sources): WhatsApp via local ChatStorage.sqlite`
- `feat(sources): GitHub via REST API`

---

### Atom 7 — Skills + docs

**What:** per-source `SKILL.md` files so Claude Code knows when to suggest each. Top-level `apps/sources/README.md` listing all sources with status. `docs/SOURCES.md` aimed at humans.

**Commit:** `docs(sources): per-source skills + catalog README`

---

### Atom 8 — Eval + cleanup + close

**What:** validate that:
- Registry discovers all migrated sources via entry-point discovery.
- Each emits SIGNALS-compliant chunks across all four temporal axes (extend `test_signals_schema.py` to iterate over all manifests).
- Existing Wave 1 + 1.5 + 2 narrow regressions still green.
- Per-source `validate` reports human-readable status (works / missing data / missing auth / wrong permissions).
- **Cleanup:** delete the old `index-<source>` implementation code from `packages/leann-core/src/leann/cli.py` and `apps/*_rag.py` once all migrations are confirmed green. Keep only the deprecated-alias shims for one release.

Done bar:
- 9 sources registered in `registry.yaml` (7 migrated + WhatsApp + GitHub).
- All pass schema validation including four-axis temporal coverage.
- Full Wave 1 + 1.5 + 2 regression slice green (current bar: 388 passed / 14 skipped / 18 deselected).
- `leann sources list` prints the catalog with category grouping.
- `leann sources validate <name>` runs successfully for every catalog entry the user has access to.

**Commit:** `docs(dev): wave 3 closed — source registry`

---

## /loop driver prompt

When ready to dispatch, paste this:

```
You are executing Wave 3 (source registry) of docs/dev/SOURCE-REGISTRY-PLAN.md.

Branch: feat/source-registry (cut from feat/multi-axis-temporal at HEAD 8199817).

Atoms 0-8 in strict order. Decisions section in the plan is LOCKED — do not
revisit (YAML manifests, new packages/leann-sources/ package, full migration
of 7 readers, WhatsApp + GitHub as new sources, single dispatch).

Required reading before each atom:
- docs/dev/SOURCE-REGISTRY-PLAN.md (the atom itself)
- docs/dev/SIGNALS.md (chunks MUST be SIGNALS-compliant across all 4 axes)
- docs/UPGRADE.md (current API surface — don't break it)

Setup invariants — halt if any fail:
- git branch = feat/source-registry
- .venv/bin/python -c "import leann, dateparser, yaml, jsonschema" succeeds
- Wave 1 + 1.5 + 2 narrow regression passes (extend the slice as new
  Wave 1.5 tests landed in feat/multi-axis-temporal — see latest
  PROGRESS.md entries for the canonical list).
- pkill -9 -f hnsw_embedding_server 2>/dev/null || true

Hard NO list:
- Breaking the existing leann build / index-* CLI surface. Atoms 4-5 must
  keep `leann index-<name>` working as deprecated aliases. Cleanup of the
  old implementation happens ONLY in atom 8.
- Renaming any existing public API (LeannBuilder, LeannSearcher,
  ReActAgent, the four temporal axes, etc.).
- Touching backends (packages/leann-backend-*).
- Touching Wave 1/1.5 temporal code or Wave 2 metadata-aware code.
- Adding non-SIGNALS metadata fields without updating SIGNALS.md first.
- Adding deps to leann-core. New deps belong in packages/leann-sources/pyproject.toml.

Per-atom workflow:
1. Read the atom + relevant SIGNALS.md sections.
2. State in one sentence what you'll change.
3. Edit + write the test file specified for the atom.
4. .venv/bin/pytest <atom test> -v
5. Run the Wave 1 + 1.5 + 2 narrow regression — must stay green.
6. .venv/bin/ruff check --fix && .venv/bin/ruff format (ruff pinned 0.12.7).
7. git add <files>; git commit with the shape:
     feat(sources): <atom title>  | test(sources): ...  | docs(sources): ...
8. Append to docs/dev/PROGRESS.md: "Atom N: <sha> — <one-line result>"
9. pkill -9 -f hnsw_embedding_server 2>/dev/null || true

Stop and ask when:
- An atom's done-test fails 3 times.
- Existing Wave 1/1.5/2 test regresses.
- Edit needs to touch a file NOT in the atom's "What" section.
- Manifest field shape needs to deviate from the locked schema example.

Acceptance bar (atom 8):
- 9 sources registered in packages/leann-sources/sources/registry.yaml.
- All sources emit SIGNALS-compliant chunks with all four temporal axes
  (created_at, modified_at, event_time, indexed_at) where applicable;
  synthesized markers present where axes are degenerate.
- leann sources list / info / validate / connect / install all work.
- leann index --source <name> works for every catalog entry the user has
  access to.
- Wave 1 + 1.5 + 2 narrow regression unchanged (388 passed / 14 skipped /
  18 deselected as of HEAD 8199817).

Tools: .venv/bin/{python,pytest,ruff}. Embedding server: iq at
http://100.122.112.83:8100/v1 (check with curl before any indexing-heavy work).
```

---

## Decisions (locked 2026-05-20)

1. **Package shape:** new package `packages/leann-sources/` parallel to backend packages. Opt-in install. **Not vendored into leann-core.** Discovered via `leann.plugins` entry points.
2. **Manifest format:** **YAML** (`.yaml`). Separate-package constraint frees us from the pyyaml-in-core concern; YAML reads materially cleaner than JSON for nested field-mapping + transforms. (PP uses JSON because their manifests are flat; ours aren't.) jsonschema validates the YAML-loaded dict.
3. **Versioning:** every manifest carries `manifest_version` (schema version, semver-major rejection on mismatch) AND `version` (source-specific). Both required.
4. **Migration scope:** **full** — all 7 hardcoded readers move into the new package. Old `leann index-<source>` commands become deprecated aliases for one release.
5. **First new sources:** **WhatsApp + GitHub**, per the original plan. Notion + Linear deferred to Wave 3.1 (need user data + auth setup decisions). GCal also Wave 3.1.
6. **Importance signals:** declare-only in Wave 3 manifests. Consumption (decay-weighted scoring, connector edges) lives in a future wave.
7. **PP-style installer:** **no.** `leann sources install` validates deps + creates config dir; it does not orchestrate npm/Go/Cargo binaries.
8. **Dispatch:** **single /loop session**, atoms 0 → 8 serial. Same shape as Wave 1 and Wave 1.5.

---

## Out of scope — for waves after this

- **Importance-aware retrieval scoring** — uses Wave 3's importance signals to weight results. Belongs in Wave 5/6 alongside temporal decay.
- **Cross-source connector layer** — uses Wave 3's `connectors.cross_source_match_by` declarations to materialize edges. Wave 6.
- **OAuth flow polish** — atom 3's `sources connect` covers the basic case; a proper credentials manager (Keychain integration, refresh-token handling) is its own mini-wave later.
- **Source-specific MCP servers** — Printing Press wraps each catalog entry as both CLI + MCP. We're not doing MCP per source initially; LEANN's existing single MCP server is enough.
- **Live-stream / push sources** — Wave 3 is pull-based. Push-based sources (webhooks, Kafka, Slack events) come later if needed.
- **Sharing/publishing manifests** — PP has `printingpress.dev`; we can publish ours later if it's worth it.
