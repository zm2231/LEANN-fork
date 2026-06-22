"""Content-addressed embedding cache.

The cache stores vectors by the exact canonical text and output-affecting
embedding configuration. Operational knobs such as batch size and timeouts are
intentionally excluded so they do not fragment cache reuse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

CACHE_SCHEMA_VERSION = 1
CANONICALIZATION_VERSION = 1
MODEL_REVISION_ENV = "LEANN_EMBED_MODEL_REV"
MODEL_REVISION_PROBE_TIMEOUT_ENV = "LEANN_EMBED_MODEL_REV_PROBE_TIMEOUT"

logger = logging.getLogger(__name__)

_model_revision_cache: dict[tuple[str, str, str, bool], str] = {}


def canonicalize_text(text: str) -> str:
    """Canonical text that is both hashed and sent to the embedder."""
    normalized = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in normalized.splitlines()]
    return "\n".join(lines)


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """Return float32 row-wise L2-normalized vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cache_enabled() -> bool:
    value = os.environ.get("LEANN_EMBED_CACHE", "").strip().lower()
    return value not in {"0", "false", "no", "off", "disabled"}


def default_cache_path() -> Path:
    configured = os.environ.get("LEANN_EMBED_CACHE_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".leann" / "embed-cache.sqlite"


def build_cache_config(
    *,
    mode: str,
    model_name: str,
    provider: str | None = None,
    dimensions: int | None = None,
    normalization: str = "none",
    pooling: str | None = None,
    truncation: dict[str, Any] | None = None,
    prompt_template: str | None = None,
    model_rev: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "mode": mode,
        "model": model_name,
        "provider": provider or "",
        "dimensions": dimensions,
        "normalization": normalization,
        "pooling": pooling or "",
        "truncation": truncation or {},
        "prompt_template": prompt_template or "",
    }
    if model_rev:
        config["model_rev"] = model_rev
    if extra:
        config["extra"] = extra
    return config


def cache_key(config: dict[str, Any], text: str) -> str:
    payload = {
        "config": config,
        "text": text,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_model_revision(
    *,
    mode: str,
    model_name: str,
    provider: str | None = None,
    api_key: str | None = None,
) -> str:
    configured = os.environ.get(MODEL_REVISION_ENV, "").strip()
    if configured:
        return configured
    if mode != "openai" or not provider:
        return ""

    cache_key = (mode, provider.rstrip("/"), model_name, bool(api_key))
    cached = _model_revision_cache.get(cache_key)
    if cached is not None:
        return cached

    revision = _probe_openai_model_revision(provider, model_name, api_key=api_key)
    _model_revision_cache[cache_key] = revision
    return revision


def _probe_openai_model_revision(provider: str, model_name: str, api_key: str | None = None) -> str:
    timeout = _model_revision_probe_timeout()
    url = f"{provider.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as exc:
        logger.debug("Could not probe embedding model revision from %s: %s", url, exc)
        return ""
    return _extract_model_revision(payload, model_name)


def _model_revision_probe_timeout() -> float:
    value = os.environ.get(MODEL_REVISION_PROBE_TIMEOUT_ENV, "").strip()
    if not value:
        return 1.0
    try:
        return max(0.05, float(value))
    except ValueError:
        return 1.0


def _extract_model_revision(payload: Any, model_name: str) -> str:
    if not isinstance(payload, dict):
        return ""
    data = payload.get("data")
    if not isinstance(data, list):
        return ""

    model = _select_model_entry(data, model_name)
    if not model:
        return ""

    explicit_fields = (
        "model_revision",
        "backend_model_revision",
        "weights_sha256",
        "model_fingerprint",
        "revision",
        "snapshot",
        "digest",
        "sha",
    )
    for field in explicit_fields:
        value = model.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value:
            return str(value)

    backend_model = model.get("backend_model")
    if backend_model:
        derived = {
            "backend_model": backend_model,
            "context_length": model.get("context_length"),
            "dimensions": model.get("dimensions"),
        }
        return json.dumps(derived, sort_keys=True, separators=(",", ":"))

    return ""


def _select_model_entry(data: list[Any], model_name: str) -> dict[str, Any] | None:
    candidates = [item for item in data if isinstance(item, dict)]
    for item in candidates:
        if item.get("id") == model_name or item.get("model") == model_name:
            return item
    return None


class EmbeddingCache:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path is not None else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              key TEXT PRIMARY KEY,
              dim INTEGER NOT NULL,
              dtype TEXT NOT NULL,
              vector BLOB NOT NULL,
              config_json TEXT NOT NULL,
              model_rev TEXT NOT NULL DEFAULT '',
              created_at REAL NOT NULL,
              last_accessed_at REAL NOT NULL,
              hits INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        columns = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "model_rev" not in columns:
            self.conn.execute("ALTER TABLE embeddings ADD COLUMN model_rev TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self.conn.close()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        hits: dict[str, np.ndarray] = {}
        unique_keys = list(dict.fromkeys(keys))
        placeholders = ",".join("?" for _ in unique_keys)
        rows = self.conn.execute(
            f"SELECT key, dim, dtype, vector FROM embeddings WHERE key IN ({placeholders})",
            unique_keys,
        ).fetchall()
        stale_keys: list[str] = []
        for key, dim, dtype, blob in rows:
            try:
                if dtype != "float32":
                    raise ValueError(f"unsupported dtype {dtype}")
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.size != int(dim):
                    raise ValueError("dimension mismatch")
                hits[str(key)] = vector.copy()
            except Exception:
                stale_keys.append(str(key))
        if stale_keys:
            self.delete_many(stale_keys)
        if hits:
            now = time.time()
            self.conn.executemany(
                "UPDATE embeddings SET last_accessed_at = ?, hits = hits + 1 WHERE key = ?",
                [(now, key) for key in hits],
            )
            self.conn.commit()
        return hits

    def put_many(
        self,
        items: list[tuple[str, np.ndarray]],
        *,
        config: dict[str, Any],
    ) -> None:
        if not items:
            return
        now = time.time()
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        model_rev = str(config.get("model_rev") or "")
        rows = []
        for key, vector in items:
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    key,
                    int(arr.size),
                    "float32",
                    arr.tobytes(),
                    config_json,
                    model_rev,
                    now,
                    now,
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
              (key, dim, dtype, vector, config_json, model_rev, created_at, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def delete_many(self, keys: list[str]) -> None:
        if not keys:
            return
        self.conn.executemany("DELETE FROM embeddings WHERE key = ?", [(key,) for key in keys])
        self.conn.commit()


def get_many(keys: list[str], path: str | Path | None = None) -> dict[str, np.ndarray]:
    cache = EmbeddingCache(path)
    try:
        return cache.get_many(keys)
    finally:
        cache.close()


def put_many(
    items: list[tuple[str, np.ndarray]],
    *,
    config: dict[str, Any],
    path: str | Path | None = None,
) -> None:
    cache = EmbeddingCache(path)
    try:
        cache.put_many(items, config=config)
    finally:
        cache.close()
