import numpy as np

from leann.api import LeannSearcher, SearchResult


class FakeBackend:
    def __init__(self):
        self.search_calls = 0

    def _ensure_server_running(self, *args, **kwargs):
        return None

    def compute_query_embedding(self, *args, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def search(self, *args, **kwargs):
        self.search_calls += 1
        return {"labels": [["ann-0", "ann-1"]], "distances": [[0.9, 0.8]]}


class FakePassageManager:
    def __init__(self, selectivity):
        self.selectivity = selectivity
        self.prefilter_calls = 0
        self.configure_calls = []
        self.passages = {
            "ann-0": {
                "text": "ann hit 0",
                "metadata": {
                    "channel": "common",
                    "source_document_id": "doc-ann",
                    "chunk_seq": 1,
                },
            },
            "ann-1": {
                "text": "ann hit 1",
                "metadata": {
                    "channel": "common",
                    "source_document_id": "doc-ann",
                    "chunk_seq": 2,
                },
            },
        }

    def __len__(self):
        return 100

    def configure_embedding_pipeline(self, *args, **kwargs):
        self.configure_calls.append((args, kwargs))

    def estimate_selectivity(self, metadata_filters):
        return self.selectivity

    def filter_stats(self, metadata_filters):
        matches = int(self.selectivity * 100)
        return {
            "total_passages": 100,
            "filter_matches": matches,
            "filter_selectivity": self.selectivity,
        }

    def score_filtered_subset(self, query_embedding, metadata_filters, top_k, exclude_ids=None):
        self.prefilter_calls += 1
        return [
            SearchResult(
                id=str(i),
                score=1.0 - (i * 0.1),
                text=f"prefilter hit {i}",
                metadata={"channel": "rare"},
            )
            for i in range(min(3, top_k))
        ]

    def get_passage(self, passage_id):
        return self.passages[passage_id]

    def fetch_siblings(self, source_document_id, chunk_seq, before, after):
        return [
            SearchResult(
                id=f"{source_document_id}:{seq}",
                score=0.0,
                text=f"context {seq}",
                metadata={"source_document_id": source_document_id, "chunk_seq": seq},
            )
            for seq in range(chunk_seq - before, chunk_seq + after + 1)
            if seq != chunk_seq and seq >= 0
        ]

    def filter_search_results(self, search_results, metadata_filters):
        return [
            result
            for result in search_results
            if result.metadata.get("channel") == metadata_filters["channel"]["=="]
        ]


def _searcher(selectivity):
    searcher = LeannSearcher.__new__(LeannSearcher)
    searcher.passage_manager = FakePassageManager(selectivity)
    searcher.backend_impl = FakeBackend()
    searcher.backend_name = "hnsw"
    searcher.embedding_model = "fake-model"
    searcher.embedding_mode = "fake-mode"
    searcher.embedding_options = {}
    searcher.meta_path_str = "/tmp/fake.meta.json"
    searcher.recompute_embeddings = False
    searcher._warmup = False
    searcher._use_daemon = False
    searcher._daemon_ttl_seconds = 0
    searcher.bm25_scorer = None
    return searcher


def test_sparse_auto_prefilter_returns_ranked_filtered_results():
    searcher = _searcher(selectivity=0.03)

    results = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="auto",
    )

    assert [result.id for result in results] == ["0", "1", "2"]
    assert searcher.passage_manager.prefilter_calls == 1
    assert searcher.backend_impl.search_calls == 0


def test_dense_auto_filter_falls_back_to_ann_path():
    searcher = _searcher(selectivity=0.60)

    results = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "common"}},
        prefilter="auto",
    )

    assert [result.id for result in results] == ["ann-0", "ann-1"]
    assert searcher.passage_manager.prefilter_calls == 0
    assert searcher.backend_impl.search_calls == 1


def test_prefilter_always_forces_prefilter_on_dense_filter():
    searcher = _searcher(selectivity=0.60)

    results = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="always",
    )

    assert [result.id for result in results] == ["0", "1", "2"]
    assert searcher.passage_manager.prefilter_calls == 1
    assert searcher.backend_impl.search_calls == 0


def test_prefilter_never_preserves_ann_postfilter_behavior():
    searcher = _searcher(selectivity=0.03)

    results = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="never",
    )

    assert results == []
    assert searcher.passage_manager.prefilter_calls == 0
    assert searcher.backend_impl.search_calls == 1


def test_no_metadata_filters_uses_ann_path():
    searcher = _searcher(selectivity=0.03)

    results = searcher.search("query", top_k=5, prefilter="always")

    assert [result.id for result in results] == ["ann-0", "ann-1"]
    assert searcher.passage_manager.prefilter_calls == 0
    assert searcher.backend_impl.search_calls == 1
