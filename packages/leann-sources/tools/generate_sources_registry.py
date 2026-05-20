#!/usr/bin/env python3
"""Regenerate packages/leann-sources/sources/registry.yaml."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))


def main() -> int:
    from leann_sources.registry import write_registry

    sources_root = PACKAGE_ROOT / "sources"
    registry = write_registry(sources_root)
    rel_path = sources_root.relative_to(REPO_ROOT) / "registry.yaml"
    print(f"wrote {rel_path} ({len(registry.entries)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
