# packages/leann-core/src/leann/index_manifest.py
"""Global index manifest — a fast cache for ``leann list``.

Background
----------
``leann list`` historically discovered indexes by ``os.walk``-ing every
registered project directory (``~/.leann/projects.json``) looking for
``*.leann.meta.json`` files. On a machine with many registered projects —
especially on a slow/external volume — that walk dominates wall time (it stats
thousands of files across 25+ trees).

This module replaces the per-list walk with a single JSON manifest at
``~/.leann/indexes.json``. Index *creation* (``LeannBuilder.build_index`` /
``build_index_from_arrays``) upserts an entry; index *removal*
(``leann remove`` → ``_delete_index_directory``) forgets it. ``leann list``
then reads the manifest and only ``stat()``-verifies each entry's meta file,
pruning entries whose files have disappeared (e.g. a manual ``rm -rf``).
``leann list`` reads the manifest first. If it is empty, the CLI seeds it from
cheap CLI-layout index directories only; ``leann list --refresh`` is the
explicit full scan for legacy app-format indexes.

Each entry is keyed by the resolved ``<name>.meta.json`` path and carries
everything ``leann list`` renders (name, type, project, size) plus build
provenance (backend, embedding model/mode, dimensions) for future use.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def manifest_path() -> Path:
    return Path.home() / ".leann" / "indexes.json"


def _key(meta_path: Path) -> str:
    try:
        return str(Path(meta_path).resolve())
    except Exception:
        return str(meta_path)


def load_manifest() -> dict[str, Any]:
    path = manifest_path()
    if not path.exists():
        return {"version": MANIFEST_VERSION, "indexes": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("indexes"), dict):
            return {"version": MANIFEST_VERSION, "indexes": {}}
        return data
    except Exception:
        logger.debug("Could not load index manifest; treating as empty")
        return {"version": MANIFEST_VERSION, "indexes": {}}


def _save_manifest(data: dict[str, Any]) -> None:
    path = manifest_path()
    try:
        path.parent.mkdir(exist_ok=True)
        # Atomic replace so a concurrent reader never sees a half-written file.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".indexes.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Could not save index manifest: {e}")


def _is_cli_layout(index_dir: Path) -> bool:
    """True for CLI-format indexes at ``<project>/.leann/indexes/<name>/``."""
    parts = index_dir.parts
    return len(parts) >= 3 and parts[-3] == ".leann" and parts[-2] == "indexes"


def _resolve(p: Path | str) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


def _project_for(index_dir: Path) -> str:
    """Best-effort project directory used to group entries in ``leann list``.

    CLI-format indexes live at ``<project>/.leann/indexes/<name>/`` → project is
    ``<project>``. App-format indexes are grouped under the nearest ancestor that
    contains a ``.leann`` directory (the registered project root), falling back
    to the index directory itself. Always returned resolved so it compares equal
    to ``Path.cwd().resolve()`` in the renderer (e.g. /tmp vs /private/tmp).
    """
    if _is_cli_layout(index_dir):
        return _resolve(index_dir.parents[2])
    cur = index_dir
    for _ in range(8):  # bounded climb to avoid walking to filesystem root
        try:
            if (cur / ".leann").exists():
                return _resolve(cur)
        except Exception:
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    return _resolve(index_dir)


def _file_base(meta_path: Path) -> str:
    name = meta_path.name
    if name.endswith(".meta.json"):
        return name[: -len(".meta.json")]
    return meta_path.stem


def _dir_size_mb(index_dir: Path, file_base: str, is_cli: bool) -> float:
    try:
        if is_cli:
            files = (f for f in index_dir.iterdir() if f.is_file())
        else:
            files = (f for f in index_dir.glob(f"{file_base}.leann*") if f.is_file())
        return sum(f.stat().st_size for f in files) / (1024 * 1024)
    except (OSError, PermissionError):
        return 0.0


def make_entry(
    meta_path: Path,
    *,
    name: str | None = None,
    project: str | None = None,
    backend: str | None = None,
    embedding_model: str | None = None,
    embedding_mode: str | None = None,
    dimensions: int | None = None,
) -> dict[str, Any]:
    meta_path = Path(meta_path)
    index_dir = meta_path.parent
    is_cli = _is_cli_layout(index_dir)
    file_base = _file_base(meta_path)
    return {
        "name": name or index_dir.name,
        "type": "cli" if is_cli else "app",
        "project": _resolve(project) if project else _project_for(index_dir),
        "index_dir": _resolve(index_dir),
        "meta_path": _key(meta_path),
        "file_base": file_base,
        "backend": backend,
        "embedding_model": embedding_model,
        "embedding_mode": embedding_mode,
        "dimensions": dimensions,
        "size_mb": _dir_size_mb(index_dir, file_base, is_cli),
    }


def record_index(meta_path: Path, **kwargs: Any) -> None:
    """Upsert a manifest entry for a freshly built/updated index. Never raises."""
    try:
        entry = make_entry(Path(meta_path), **kwargs)
        data = load_manifest()
        data["indexes"][_key(meta_path)] = entry
        _save_manifest(data)
    except Exception as e:
        logger.debug(f"record_index failed (non-fatal): {e}")


def forget_index(meta_path: Path) -> None:
    """Drop the manifest entry for a specific meta file. Never raises."""
    try:
        data = load_manifest()
        if data["indexes"].pop(_key(meta_path), None) is not None:
            _save_manifest(data)
    except Exception as e:
        logger.debug(f"forget_index failed (non-fatal): {e}")


def forget_dir(index_dir: Path) -> None:
    """Drop every manifest entry living at or under ``index_dir``. Never raises."""
    try:
        target = str(Path(index_dir).resolve())
        data = load_manifest()
        drop = []
        for key, entry in data["indexes"].items():
            ent_dir = entry.get("index_dir", "")
            try:
                ent_dir_res = str(Path(ent_dir).resolve())
            except Exception:
                ent_dir_res = ent_dir
            if ent_dir_res == target or ent_dir_res.startswith(target + os.sep):
                drop.append(key)
        if drop:
            for k in drop:
                data["indexes"].pop(k, None)
            _save_manifest(data)
    except Exception as e:
        logger.debug(f"forget_dir failed (non-fatal): {e}")


def iter_indexes(verify: bool = True) -> Iterator[dict[str, Any]]:
    """Yield manifest entries, pruning any whose meta file no longer exists."""
    data = load_manifest()
    stale: list[str] = []
    for key, entry in list(data["indexes"].items()):
        if verify:
            mp = entry.get("meta_path") or key
            if not Path(mp).exists():
                stale.append(key)
                continue
        yield entry
    if verify and stale:
        for k in stale:
            data["indexes"].pop(k, None)
        _save_manifest(data)


def replace_all(entries: list[dict[str, Any]]) -> None:
    """Rebuild the manifest from a full set of entries (used by ``--refresh``)."""
    data = {"version": MANIFEST_VERSION, "indexes": {}}
    for entry in entries:
        key = entry.get("meta_path")
        if not key:
            continue
        data["indexes"][_key(Path(key))] = entry
    _save_manifest(data)
