#!/usr/bin/env python3
"""Verify that the active Python imports LEANN from this checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import re
from pathlib import Path

PACKAGE_SPECS = {
    "leann-core": ("leann", "packages/leann-core/pyproject.toml", "packages/leann-core/src/leann"),
    "leann-backend-hnsw": (
        "leann_backend_hnsw",
        "packages/leann-backend-hnsw/pyproject.toml",
        "packages/leann-backend-hnsw/leann_backend_hnsw",
    ),
    "leann-backend-ivf": (
        "leann_backend_ivf",
        "packages/leann-backend-ivf/pyproject.toml",
        "packages/leann-backend-ivf/leann_backend_ivf",
    ),
    "leann-backend-flat": (
        "leann_backend_flat",
        "packages/leann-backend-flat/pyproject.toml",
        "packages/leann-backend-flat/leann_backend_flat",
    ),
    "leann-backend-diskann": (
        "leann_backend_diskann",
        "packages/leann-backend-diskann/pyproject.toml",
        "packages/leann-backend-diskann/leann_backend_diskann",
    ),
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_pyproject_version(pyproject_path: Path) -> str:
    text = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', text)
    if not match:
        raise AssertionError(f"version field not found in {pyproject_path}")
    return match.group(1)


def digest_tree(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def assert_package_version(dist_name: str, expected: str) -> None:
    installed = importlib.metadata.version(dist_name)
    if installed != expected:
        raise AssertionError(
            f"{dist_name} version mismatch: installed={installed} expected={expected}"
        )


def assert_tree_matches(module_name: str, repo_package_root: Path) -> Path:
    module = importlib.import_module(module_name)
    installed_root = Path(module.__file__).resolve().parent
    repo_hashes = digest_tree(repo_package_root)
    installed_hashes = digest_tree(installed_root)
    missing = sorted(set(repo_hashes) - set(installed_hashes))[:8]
    changed = sorted(
        rel
        for rel in set(repo_hashes) & set(installed_hashes)
        if repo_hashes[rel] != installed_hashes[rel]
    )[:8]
    if missing or changed:
        extra = sorted(set(installed_hashes) - set(repo_hashes))[:8]
        raise AssertionError(
            f"{module_name} installed source does not match checkout\n"
            f"installed={installed_root}\n"
            f"repo={repo_package_root}\n"
            f"missing={missing}\n"
            f"extra={extra}\n"
            f"changed={changed}"
        )
    return installed_root


def assert_hnsw_stored_vector_scoring() -> None:
    from leann_backend_hnsw.hnsw_backend import HNSWSearcher

    if not hasattr(HNSWSearcher, "score_passage_ids"):
        raise AssertionError("HNSWSearcher.score_passage_ids missing")
    if not hasattr(HNSWSearcher, "supports_stored_vector_scoring"):
        raise AssertionError("HNSWSearcher.supports_stored_vector_scoring missing")


def verify_install(repo_root: Path) -> None:
    versions = {
        dist_name: read_pyproject_version(repo_root / pyproject_rel)
        for dist_name, (_, pyproject_rel, _) in PACKAGE_SPECS.items()
    }
    for dist_name, expected in versions.items():
        assert_package_version(dist_name, expected)
    print(
        "versions=ok " + " ".join(f"{name}={version}" for name, version in sorted(versions.items()))
    )

    imported_roots = {}
    for dist_name, (module_name, _, package_rel) in PACKAGE_SPECS.items():
        imported_roots[dist_name] = assert_tree_matches(module_name, repo_root / package_rel)
    print("source_hashes=ok")
    for dist_name, imported_root in sorted(imported_roots.items()):
        print(f"{dist_name}={imported_root}")

    assert_hnsw_stored_vector_scoring()
    print("stored_vector_scoring=ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ["REPO_ROOT"]).resolve()
        if "REPO_ROOT" in os.environ
        else repo_root_from_script(),
        help="LEANN checkout root. Defaults to REPO_ROOT or the parent of this script directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_install(args.repo_root.resolve())


if __name__ == "__main__":
    main()
