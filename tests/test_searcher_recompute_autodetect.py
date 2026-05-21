"""LeannSearcher auto-detects is_recompute from meta.json when not explicit.

Fix lands the no-recompute path silently when the caller forgets to pass
recompute_embeddings=False. Was the root of the Rubio agent's "stored-vector
prefilter not firing" surprise.
"""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_minimal_index(meta_overrides: dict) -> Path:
    """Write just enough on-disk artifacts that LeannSearcher.__init__ can
    parse meta.json without actually loading a backend.
    """
    tmp = Path(tempfile.mkdtemp())
    base = tmp / "idx.leann"
    meta = {
        "version": "1.0",
        "backend_name": "hnsw",
        "embedding_model": "fake-model",
        "dimensions": 8,
        "embedding_mode": "sentence-transformers",
        "backend_kwargs": {
            "graph_degree": 32,
            "complexity": 64,
            "is_compact": False,
            "is_recompute": True,
        },
        "passage_sources": [],
        "is_compact": False,
        "is_pruned": False,
    }
    # Apply overrides (e.g., flip is_recompute to False)
    for k, v in meta_overrides.items():
        if "." in k:
            top, sub = k.split(".", 1)
            meta[top][sub] = v
        else:
            meta[k] = v
    Path(f"{base}.meta.json").write_text(json.dumps(meta))
    return base


class _StubBackendImpl:
    """Minimal stand-in for HNSWSearcher; auto-detect runs before any usage."""
    def __init__(self, *a, **kw):
        pass


class _StubBackend:
    @staticmethod
    def searcher(*a, **kw):
        return _StubBackendImpl()


def _make_searcher(meta_overrides, **searcher_kwargs):
    """Construct a LeannSearcher far enough to inspect self.recompute_embeddings.

    Stubs out the backend factory so we don't need a real HNSW index on disk.
    """
    from leann.api import LeannSearcher

    base = _write_minimal_index(meta_overrides)
    with patch.dict("leann.api.BACKEND_REGISTRY", {"hnsw": _StubBackend}):
        return LeannSearcher(str(base), enable_warmup=False, **searcher_kwargs)


def test_searcher_autodetects_recompute_false_from_meta():
    s = _make_searcher({"backend_kwargs.is_recompute": False})
    assert s.recompute_embeddings is False


def test_searcher_autodetects_recompute_true_from_meta():
    s = _make_searcher({"backend_kwargs.is_recompute": True})
    assert s.recompute_embeddings is True


def test_searcher_explicit_recompute_false_overrides_meta_true():
    s = _make_searcher(
        {"backend_kwargs.is_recompute": True},
        recompute_embeddings=False,
    )
    assert s.recompute_embeddings is False


def test_searcher_explicit_recompute_true_overrides_meta_false():
    s = _make_searcher(
        {"backend_kwargs.is_recompute": False},
        recompute_embeddings=True,
    )
    assert s.recompute_embeddings is True


def test_searcher_missing_backend_kwargs_defaults_to_recompute_true():
    """Old indexes without backend_kwargs in meta default to safe-side True."""
    base = _write_minimal_index({})
    # Strip backend_kwargs entirely
    meta_path = Path(f"{base}.meta.json")
    meta = json.loads(meta_path.read_text())
    del meta["backend_kwargs"]
    meta_path.write_text(json.dumps(meta))

    from leann.api import LeannSearcher
    with patch.dict("leann.api.BACKEND_REGISTRY", {"hnsw": _StubBackend}):
        s = LeannSearcher(str(base), enable_warmup=False)
    assert s.recompute_embeddings is True
