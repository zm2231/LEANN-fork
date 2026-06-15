from leann.embedding_server_manager import EmbeddingServerManager


def _config_signature():
    return {
        "model_name": "BAAI/bge-m3",
        "passages_file": "/tmp/example.meta.json",
        "embedding_mode": "sentence-transformers",
        "distance_metric": "mips",
        "provider_options": {},
        "passages_signature": None,
    }


def test_server_info_matches_expected_identity():
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    config = _config_signature()
    info = {
        "protocol": "leann-embedding-server",
        "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
        "pid": 123,
        "model_name": "BAAI/bge-m3",
        "passages_file": "/tmp/example.meta.json",
        "embedding_mode": "sentence-transformers",
        "distance_metric": "mips",
    }

    assert manager._server_info_matches(info, config, expected_pid=123)


def test_server_info_rejects_different_model():
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    config = _config_signature()
    info = {
        "protocol": "leann-embedding-server",
        "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
        "pid": 123,
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "passages_file": "/tmp/example.meta.json",
        "embedding_mode": "sentence-transformers",
        "distance_metric": "mips",
    }

    assert not manager._server_info_matches(info, config, expected_pid=123)


def test_server_info_rejects_different_pid_when_expected():
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    config = _config_signature()
    info = {
        "protocol": "leann-embedding-server",
        "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
        "pid": 999,
        "model_name": "BAAI/bge-m3",
        "passages_file": "/tmp/example.meta.json",
        "embedding_mode": "sentence-transformers",
        "distance_metric": "mips",
    }

    assert not manager._server_info_matches(info, config, expected_pid=123)


def test_adopt_reuses_exact_registry_match_when_identity_probe_times_out(tmp_path, monkeypatch):
    manager = EmbeddingServerManager("leann_backend_hnsw.hnsw_embedding_server")
    config = _config_signature()
    registry_dir = tmp_path / "servers"
    registry_dir.mkdir()
    registry_path = registry_dir / f"{manager._registry_key(config)}.json"
    registry_path.write_text(
        """{
  "pid": 123,
  "port": 5557,
  "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
  "daemon_ttl_seconds": 900,
  "created_at": 1,
  "config_signature": {
    "model_name": "BAAI/bge-m3",
    "passages_file": "/tmp/example.meta.json",
    "embedding_mode": "sentence-transformers",
    "distance_metric": "mips",
    "provider_options": {},
    "passages_signature": null
  }
}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(EmbeddingServerManager, "_registry_dir", staticmethod(lambda: registry_dir))
    monkeypatch.setattr("leann.embedding_server_manager._pid_is_alive", lambda pid: True)
    monkeypatch.setattr("leann.embedding_server_manager._check_port", lambda port: True)
    monkeypatch.setattr(manager, "_query_server_info", lambda port: None)

    assert manager._adopt_registered_server(config) == 5557
    assert registry_path.exists()
