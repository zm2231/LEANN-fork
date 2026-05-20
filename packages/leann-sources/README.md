# LEANN Sources

`leann-sources` is the optional source-registry package for LEANN. It contains
declarative source manifests, shared reader base classes, manifest validation,
and the plugin entry point that later Wave 3 atoms wire into `leann-core`.

Install it for local development with:

```bash
uv pip install -e packages/leann-sources/
```

Source manifests live under `packages/leann-sources/sources/`. Each manifest
declares where the raw data lives, how to validate access, how source fields map
to SIGNALS metadata, and which reader class can emit LEANN chunks.
