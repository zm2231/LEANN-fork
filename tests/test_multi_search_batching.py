from __future__ import annotations

from types import MethodType

import numpy as np
from leann.api import LeannSearcher


class RecordingBackend:
    def __init__(self) -> None:
        self.batch_calls = []

    def compute_query_embeddings(self, queries, **kwargs):
        self.batch_calls.append((list(queries), kwargs))
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def make_searcher() -> LeannSearcher:
    searcher = LeannSearcher.__new__(LeannSearcher)
    searcher.recompute_embeddings = False
    searcher.embedding_options = {}
    searcher.backend_impl = RecordingBackend()
    searcher.search_calls = []

    def record_search(self, query, **kwargs):
        query_embedding = kwargs.get("query_embedding")
        self.search_calls.append(
            (
                query,
                None if query_embedding is None else query_embedding.tolist(),
                {key: value for key, value in kwargs.items() if key != "query_embedding"},
            )
        )
        return [query]

    searcher.search = MethodType(record_search, searcher)
    return searcher


def test_multi_search_batches_query_embeddings_for_vector_search() -> None:
    searcher = make_searcher()

    results = searcher.multi_search(["q1", "q2"], top_k=3, vector_weight=0.5)

    assert results == [["q1"], ["q2"]]
    assert searcher.backend_impl.batch_calls == [
        (["q1", "q2"], {"use_server_if_available": False, "query_template": None})
    ]
    assert searcher.search_calls == [
        ("q1", [[1.0, 0.0]], {"top_k": 3, "provider_options": None, "vector_weight": 0.5}),
        ("q2", [[0.0, 1.0]], {"top_k": 3, "provider_options": None, "vector_weight": 0.5}),
    ]


def test_multi_search_uses_prompt_template_for_batched_embeddings() -> None:
    searcher = make_searcher()

    searcher.multi_search(["q1", "q2"], provider_options={"prompt_template": "query: {}"})

    assert searcher.backend_impl.batch_calls == [
        (["q1", "q2"], {"use_server_if_available": False, "query_template": "query: {}"})
    ]


def test_multi_search_falls_back_when_embeddings_are_not_reused() -> None:
    fallback_kwargs = [
        {"vector_weight": 0.0},
        {"enable_temporal": True},
        {"use_grep": True},
    ]
    for kwargs in fallback_kwargs:
        searcher = make_searcher()

        results = searcher.multi_search(["q1", "q2"], top_k=2, **kwargs)

        assert results == [["q1"], ["q2"]]
        assert searcher.backend_impl.batch_calls == []
        assert searcher.search_calls == [
            ("q1", None, {"top_k": 2, "provider_options": None, **kwargs}),
            ("q2", None, {"top_k": 2, "provider_options": None, **kwargs}),
        ]


def test_multi_search_falls_back_for_recompute_mode() -> None:
    searcher = make_searcher()
    searcher.recompute_embeddings = True

    results = searcher.multi_search(["q1", "q2"], top_k=2)

    assert results == [["q1"], ["q2"]]
    assert searcher.backend_impl.batch_calls == []
    assert searcher.search_calls == [
        ("q1", None, {"top_k": 2, "provider_options": None}),
        ("q2", None, {"top_k": 2, "provider_options": None}),
    ]
