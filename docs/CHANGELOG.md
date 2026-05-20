# Changelog

## Unreleased

### Added

- Wave 1.5 multi-axis temporal search:
  - SIGNALS now defines `created_at`, `modified_at`, `event_time`, `indexed_at`, and temporal-axis diagnostics.
  - Filesystem and specialized readers emit available temporal axes while omitting meaningless `event_time` values.
  - Natural-language temporal search routes to the appropriate axis, supports `temporal_axis` overrides, and supports `temporal_strict=True` for no-fallback behavior.
  - `scripts/build_context_layer_temporal.py` builds a context-layer temporal eval corpus and `scripts/eval_temporal.py --multi-axis` evaluates Wave 1 and Wave 1.5 gold sets together.
