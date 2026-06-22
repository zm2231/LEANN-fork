import sqlite3

import numpy as np


def test_embedding_cache_config_omits_empty_model_revision():
    from leann.embedding_cache import build_cache_config, cache_key

    config = build_cache_config(
        mode="openai",
        model_name="BAAI/bge-m3",
        provider="http://iq.example/v1",
        dimensions=1024,
        model_rev="",
    )

    assert "model_rev" not in config
    assert cache_key(config, "hello") == cache_key(dict(config), "hello")


def test_build_embedding_cache_reuses_canonical_text(monkeypatch, tmp_path):
    from leann.embedding_compute import compute_embeddings

    calls: list[list[str]] = []

    def fake_sentence_transformers(texts, *args, is_build=False, **kwargs):
        if is_build:
            calls.append(list(texts))
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(tmp_path / "embed-cache.sqlite"))
    monkeypatch.setattr(
        "leann.embedding_compute.compute_embeddings_sentence_transformers",
        fake_sentence_transformers,
    )

    provider_options = {"_leann_dimensions": 2}
    first = compute_embeddings(
        ["hello  ", "hello"],
        "fake-model",
        is_build=True,
        provider_options=provider_options,
    )
    second = compute_embeddings(
        ["hello"],
        "fake-model",
        is_build=True,
        provider_options=provider_options,
    )

    assert calls == [["hello"]]
    np.testing.assert_array_equal(first[0], first[1])
    np.testing.assert_array_equal(second[0], first[0])


def test_embedding_cache_key_ignores_batch_size_and_separates_dimensions(monkeypatch, tmp_path):
    from leann.embedding_compute import compute_embeddings

    calls: list[list[str]] = []

    def fake_sentence_transformers(texts, *args, is_build=False, **kwargs):
        if is_build:
            calls.append(list(texts))
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(tmp_path / "embed-cache.sqlite"))
    monkeypatch.setattr(
        "leann.embedding_compute.compute_embeddings_sentence_transformers",
        fake_sentence_transformers,
    )

    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={"_leann_dimensions": 2, "batch_size": 1},
    )
    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={
            "_leann_dimensions": 2,
            "batch_size": 64,
            "prompt_template": "not-applied: ",
        },
    )
    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={"_leann_dimensions": 3, "batch_size": 64},
    )

    assert calls == [["same text"], ["same text"]]


def test_embedding_cache_key_separates_explicit_model_revision(monkeypatch, tmp_path):
    from leann.embedding_compute import compute_embeddings

    calls: list[list[str]] = []

    def fake_sentence_transformers(texts, *args, is_build=False, **kwargs):
        if is_build:
            calls.append(list(texts))
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(tmp_path / "embed-cache.sqlite"))
    monkeypatch.setattr(
        "leann.embedding_compute.compute_embeddings_sentence_transformers",
        fake_sentence_transformers,
    )

    monkeypatch.setenv("LEANN_EMBED_MODEL_REV", "rev-a")
    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={"_leann_dimensions": 2},
    )
    monkeypatch.setenv("LEANN_EMBED_MODEL_REV", "rev-b")
    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={"_leann_dimensions": 2},
    )

    assert calls == [["same text"], ["same text"]]


def test_model_revision_env_override_beats_probe(monkeypatch):
    from leann.embedding_cache import resolve_model_revision

    def fail_probe(*args, **kwargs):
        raise AssertionError("probe should not run when env override is set")

    monkeypatch.setenv("LEANN_EMBED_MODEL_REV", "operator-rev")
    monkeypatch.setattr("leann.embedding_cache._probe_openai_model_revision", fail_probe)

    assert (
        resolve_model_revision(
            mode="openai",
            model_name="BAAI/bge-m3",
            provider="http://iq.example/v1",
        )
        == "operator-rev"
    )


def test_openai_model_revision_probe_is_memoized_and_affects_cache(monkeypatch, tmp_path):
    from leann import embedding_cache
    from leann.embedding_compute import compute_embeddings

    calls = {"probe": 0, "embed": 0}

    def fake_probe(provider, model_name, api_key=None):
        calls["probe"] += 1
        return "probe-rev"

    def fake_openai(texts, *args, is_build=False, **kwargs):
        calls["embed"] += 1
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    monkeypatch.delenv("LEANN_EMBED_MODEL_REV", raising=False)
    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(tmp_path / "embed-cache.sqlite"))
    embedding_cache._model_revision_cache.clear()
    monkeypatch.setattr("leann.embedding_cache._probe_openai_model_revision", fake_probe)
    monkeypatch.setattr("leann.embedding_compute.compute_embeddings_openai", fake_openai)

    provider_options = {
        "base_url": "http://iq.example/v1",
        "api_key": "iq-local",
        "_leann_dimensions": 2,
        "batch_size": 1,
    }
    first = compute_embeddings(
        ["alpha", "beta"],
        "BAAI/bge-m3",
        mode="openai",
        is_build=True,
        provider_options=provider_options,
    )
    second = compute_embeddings(
        ["alpha", "beta"],
        "BAAI/bge-m3",
        mode="openai",
        is_build=True,
        provider_options=provider_options,
    )

    assert calls == {"probe": 1, "embed": 1}
    np.testing.assert_array_equal(first, second)


def test_model_revision_probe_is_build_cache_only(monkeypatch):
    from leann.embedding_compute import compute_embeddings

    def fail_probe(*args, **kwargs):
        raise AssertionError("probe should not run outside build cache use")

    def fake_openai(texts, *args, is_build=False, **kwargs):
        return np.array([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    monkeypatch.delenv("LEANN_EMBED_MODEL_REV", raising=False)
    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setattr("leann.embedding_cache._probe_openai_model_revision", fail_probe)
    monkeypatch.setattr("leann.embedding_compute.compute_embeddings_openai", fake_openai)

    result = compute_embeddings(
        ["query text"],
        "BAAI/bge-m3",
        mode="openai",
        is_build=False,
        provider_options={
            "base_url": "http://iq.example/v1",
            "api_key": "iq-local",
            "_leann_dimensions": 2,
        },
    )

    np.testing.assert_array_equal(result, np.array([[10.0, 1.0]], dtype=np.float32))


def test_model_revision_probe_failure_falls_back_to_empty(monkeypatch):
    from leann import embedding_cache
    from leann.embedding_cache import resolve_model_revision

    monkeypatch.delenv("LEANN_EMBED_MODEL_REV", raising=False)
    embedding_cache._model_revision_cache.clear()
    monkeypatch.setattr("leann.embedding_cache._probe_openai_model_revision", lambda *a, **k: "")

    assert (
        resolve_model_revision(
            mode="openai",
            model_name="BAAI/bge-m3",
            provider="http://iq.example/v1",
        )
        == ""
    )


def test_extract_model_revision_uses_explicit_revision_and_backend_model():
    from leann.embedding_cache import _extract_model_revision

    explicit_payload = {
        "data": [
            {
                "id": "BAAI/bge-m3",
                "model_revision": "weights-123",
                "dimensions": 1024,
                "context_length": 8192,
            }
        ]
    }
    backend_payload = {
        "data": [
            {
                "id": "Qwen/Qwen3-Embedding-0.6B",
                "backend_model": "mlx-community/Qwen3-Embedding-0.6B-mxfp8",
                "dimensions": 1024,
                "context_length": 32768,
            }
        ]
    }
    no_revision_payload = {
        "data": [
            {
                "id": "BAAI/bge-m3",
                "provider": "infinity",
                "dimensions": 1024,
                "context_length": 8192,
            }
        ]
    }

    assert _extract_model_revision(explicit_payload, "BAAI/bge-m3") == "weights-123"
    assert "Qwen3-Embedding-0.6B-mxfp8" in _extract_model_revision(
        backend_payload, "Qwen/Qwen3-Embedding-0.6B"
    )
    assert _extract_model_revision(no_revision_payload, "BAAI/bge-m3") == ""


def test_embedding_cache_stores_model_revision_metadata(monkeypatch, tmp_path):
    from leann.embedding_compute import compute_embeddings

    def fake_sentence_transformers(texts, *args, is_build=False, **kwargs):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    cache_path = tmp_path / "embed-cache.sqlite"
    monkeypatch.setenv("LEANN_EMBED_CACHE", "1")
    monkeypatch.setenv("LEANN_EMBED_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("LEANN_EMBED_MODEL_REV", "stored-rev")
    monkeypatch.setattr(
        "leann.embedding_compute.compute_embeddings_sentence_transformers",
        fake_sentence_transformers,
    )

    compute_embeddings(
        ["same text"],
        "fake-model",
        is_build=True,
        provider_options={"_leann_dimensions": 2},
    )

    with sqlite3.connect(cache_path) as conn:
        rows = conn.execute("SELECT model_rev FROM embeddings").fetchall()
    assert rows == [("stored-rev",)]


def test_embedding_cache_migrates_existing_database_without_model_revision(tmp_path):
    from leann.embedding_cache import EmbeddingCache

    cache_path = tmp_path / "embed-cache.sqlite"
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            """
            CREATE TABLE embeddings (
              key TEXT PRIMARY KEY,
              dim INTEGER NOT NULL,
              dtype TEXT NOT NULL,
              vector BLOB NOT NULL,
              config_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              last_accessed_at REAL NOT NULL,
              hits INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()

    cache = EmbeddingCache(cache_path)
    try:
        columns = {
            row[1] for row in cache.conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        assert "model_rev" in columns
    finally:
        cache.close()


def test_embedding_cache_self_heals_dimension_mismatch(tmp_path):
    from leann.embedding_cache import EmbeddingCache

    cache_path = tmp_path / "embed-cache.sqlite"
    cache = EmbeddingCache(cache_path)
    try:
        cache.put_many(
            [("key", np.array([1.0, 2.0], dtype=np.float32))],
            config={"model": "fake"},
        )
    finally:
        cache.close()

    with sqlite3.connect(cache_path) as conn:
        conn.execute("UPDATE embeddings SET dim = 3 WHERE key = 'key'")
        conn.commit()

    cache = EmbeddingCache(cache_path)
    try:
        assert cache.get_many(["key"]) == {}
    finally:
        cache.close()
