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
