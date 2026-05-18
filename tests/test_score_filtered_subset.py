import json
import pickle

import numpy as np

from leann.api import PassageManager


def _write_passages(tmp_path, passages):
    passage_path = tmp_path / "test.passages.jsonl"
    index_path = tmp_path / "test.passages.idx"
    offset_map = {}

    with passage_path.open("w", encoding="utf-8") as f:
        for passage in passages:
            offset_map[passage["id"]] = f.tell()
            f.write(json.dumps(passage) + "\n")

    with index_path.open("wb") as f:
        pickle.dump(offset_map, f)

    manager = PassageManager(
        [
            {
                "type": "jsonl",
                "path": str(passage_path),
                "index_path": str(index_path),
            }
        ]
    )
    manager.configure_embedding_pipeline("fake-model", "fake-mode")
    return manager


def _patch_embeddings(monkeypatch, vectors_by_text):
    def fake_compute_embeddings(chunks, *args, **kwargs):
        return np.array([vectors_by_text[text] for text in chunks], dtype=np.float32)

    monkeypatch.setattr("leann.api.compute_embeddings", fake_compute_embeddings)


def _sample_passages():
    return [
        {"id": "0", "text": "alpha", "metadata": {"channel": "rare", "kind": "memo"}},
        {"id": "1", "text": "beta", "metadata": {"channel": "rare", "kind": "note"}},
        {"id": "2", "text": "gamma", "metadata": {"channel": "rare", "kind": "memo"}},
        {"id": "3", "text": "delta", "metadata": {"channel": "common", "kind": "note"}},
        {"id": "4", "text": "epsilon", "metadata": {"channel": "common", "kind": "memo"}},
        {"id": "5", "text": "zeta", "metadata": {"channel": "common", "kind": "note"}},
        {"id": "6", "text": "eta", "metadata": {"channel": "common", "kind": "memo"}},
        {"id": "7", "text": "theta", "metadata": {"channel": "common", "kind": "note"}},
        {"id": "8", "text": "iota", "metadata": {"channel": "common", "kind": "memo"}},
        {"id": "9", "text": "kappa", "metadata": {"channel": "common", "kind": "note"}},
    ]


def _sample_vectors():
    return {
        "alpha": [1.0, 0.0],
        "beta": [0.4, 0.916515],
        "gamma": [-0.2, 0.979796],
        "delta": [0.0, 1.0],
        "epsilon": [0.1, 0.994987],
        "zeta": [0.2, 0.979796],
        "eta": [0.3, 0.953939],
        "theta": [0.5, 0.866025],
        "iota": [0.6, 0.8],
        "kappa": [0.7, 0.714143],
    }


def test_score_filtered_subset_returns_matches_ranked_by_similarity(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    results = manager.score_filtered_subset(
        np.array([[1.0, 0.0]], dtype=np.float32),
        {"channel": {"==": "rare"}},
        top_k=10,
    )

    assert [result.id for result in results] == ["0", "1", "2"]
    assert [result.score for result in results] == sorted(
        [result.score for result in results], reverse=True
    )


def test_score_filtered_subset_returns_empty_when_filter_matches_zero(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    assert (
        manager.score_filtered_subset(
            np.array([[1.0, 0.0]], dtype=np.float32),
            {"channel": {"==": "missing"}},
            top_k=5,
        )
        == []
    )


def test_score_filtered_subset_respects_top_k(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    results = manager.score_filtered_subset(
        np.array([[0.0, 1.0]], dtype=np.float32),
        {"channel": {"==": "common"}},
        top_k=2,
    )

    assert len(results) == 2
    assert [result.id for result in results] == ["3", "4"]


def test_score_filtered_subset_skips_excluded_ids(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    results = manager.score_filtered_subset(
        np.array([[1.0, 0.0]], dtype=np.float32),
        {"channel": {"in": ["rare", "common"]}},
        top_k=3,
        exclude_ids={"0", "1"},
    )

    assert [result.id for result in results] == ["9", "8", "7"]


def test_score_filtered_subset_supports_list_operator(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    results = manager.score_filtered_subset(
        np.array([[1.0, 0.0]], dtype=np.float32),
        {"kind": {"in": ["memo"]}},
        top_k=10,
    )

    assert {result.metadata["kind"] for result in results} == {"memo"}
    assert {result.id for result in results} == {"0", "2", "4", "6", "8"}


def test_score_filtered_subset_scores_are_real_similarity_values(tmp_path, monkeypatch):
    manager = _write_passages(tmp_path, _sample_passages())
    _patch_embeddings(monkeypatch, _sample_vectors())

    results = manager.score_filtered_subset(
        np.array([[1.0, 0.0]], dtype=np.float32),
        {"channel": {"==": "rare"}},
        top_k=3,
    )

    assert all(isinstance(result.score, float) for result in results)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
    assert all(result.score != float("-inf") for result in results)
