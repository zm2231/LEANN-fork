# Source Manifest Schema

Every source lives under `packages/leann-sources/sources/<category>/<name>/`
and provides a `manifest.yaml`. The manifest is the contract between source
data and LEANN's canonical SIGNALS metadata.

Required top-level fields:

- `name`, `category`, `display_name`, `version`, `manifest_version`
- `data`: raw source location and access type
- `fields`: mapping from source fields to SIGNALS keys
- `chunking`: chunk granularity and token limits
- `privacy`: context-layer privacy tier

## Temporal Axes

Source manifests map source timestamps onto the four Wave 1.5 temporal axes:

- `created_at`: when the underlying item was authored or created
- `modified_at`: when the source item was last changed
- `event_time`: when the semantic event happened
- `indexed_at`: stamped by LEANN during build, not declared by readers

When a source cannot distinguish two axes, map the degenerate axis and include
`synthesized_from`. Emitted chunks must then include `<axis>_synthesized: true`
for the copied axis. For example, message sources often map `created_at` from
`event_time` and mark `created_at_synthesized`.

Timestamps must be UTC ISO 8601 strings with offsets. On macOS filesystem
sources, creation time must come from `st_birthtime`, never `st_ctime`.

## Field Mappings

Each entry in `fields` must target a reserved SIGNALS field documented in
`docs/dev/SIGNALS.md`. Source-specific data belongs under `extra`, not as a new
top-level metadata key.

Supported mapping keys:

- `source`: source column, JSON key, API property, or filesystem attribute
- `value`: constant value
- `auto`: builder-provided value such as `row_index_within_source_document_id`
- `transform`: canonical transform name
- `required`: whether missing source data is a validation error
- `synthesized_from`: temporal axis copied into this field
- `extraction` and `pattern`: regex extraction for referential signals

Canonical transforms in Atom 0:

- `unix_to_utc_iso`
- `core_data_epoch_to_utc_iso`
- `webkit_epoch_to_utc_iso`
- `regex_extract`
- `format('prefix_{value}')`
- `lookup(table.column)`
- `list_of_handles`
