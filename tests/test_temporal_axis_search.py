from datetime import datetime, timezone

import numpy as np
from leann.api import LeannSearcher, SearchResult
from leann.metadata_filter import MetadataFilterEngine

NOW = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)


class FakeBackend:
    def compute_query_embedding(self, *args, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    def search(self, *args, **kwargs):
        return {
            "labels": [["created", "event", "modified", "outside"]],
            "distances": [[0.9, 0.8, 0.7, 0.6]],
        }


class FakePassageManager:
    def __init__(self):
        self.filter_engine = MetadataFilterEngine()
        self.passages = {
            "created": {
                "text": "created hit",
                "metadata": {"created_at": "2026-05-12T00:00:00+00:00"},
            },
            "event": {
                "text": "event fallback hit",
                "metadata": {
                    "event_time": "2026-05-13T00:00:00+00:00",
                    "event_time_synthesized": True,
                },
            },
            "modified": {
                "text": "modified fallback hit",
                "metadata": {"modified_at": "2026-05-14T00:00:00+00:00"},
            },
            "outside": {
                "text": "outside hit",
                "metadata": {"created_at": "2026-04-01T00:00:00+00:00"},
            },
        }

    def __len__(self):
        return len(self.passages)

    def configure_embedding_pipeline(self, *args, **kwargs):
        return None

    def get_passage(self, passage_id):
        return self.passages[passage_id]

    def filter_search_results(self, search_results, metadata_filters):
        result_dicts = [
            {
                "id": result.id,
                "score": result.score,
                "text": result.text,
                "metadata": result.metadata,
            }
            for result in search_results
        ]
        filtered = self.filter_engine.apply_filters(result_dicts, metadata_filters)
        return [
            SearchResult(
                id=result["id"],
                score=result["score"],
                text=result["text"],
                metadata=result["metadata"],
            )
            for result in filtered
        ]

    def filter_stats(self, metadata_filters):
        result_dicts = [
            {"id": key, "text": value["text"], "metadata": value["metadata"]}
            for key, value in self.passages.items()
        ]
        matches = len(self.filter_engine.apply_filters(result_dicts, metadata_filters))
        return {
            "total_passages": len(self.passages),
            "filter_matches": matches,
            "filter_selectivity": matches / len(self.passages),
        }


def _searcher():
    searcher = LeannSearcher.__new__(LeannSearcher)
    searcher.passage_manager = FakePassageManager()
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


def test_temporal_axis_fallback_matches_adjacent_axes_and_reports_diagnostics():
    results, diagnostics = _searcher().search(
        "docs created last week",
        top_k=4,
        enable_temporal=True,
        temporal_now=NOW,
        prefilter="never",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["created", "event", "modified"]
    assert diagnostics["temporal_axis_routed"] == "created_at"
    assert diagnostics["temporal_axis_fallback_used"] == [
        ("event", "event_time"),
        ("modified", "modified_at"),
    ]
    assert diagnostics["temporal_strict"] is False
    assert diagnostics["temporal_synthesized_axes"] == 1


def test_temporal_strict_requires_routed_axis():
    results, diagnostics = _searcher().search(
        "docs created last week",
        top_k=4,
        enable_temporal=True,
        temporal_strict=True,
        temporal_now=NOW,
        prefilter="never",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["created"]
    assert diagnostics["temporal_axis_fallback_used"] == []
    assert diagnostics["temporal_strict"] is True


def test_temporal_axis_override_bypasses_parser_routing():
    results, diagnostics = _searcher().search(
        "docs changed last week",
        top_k=4,
        enable_temporal=True,
        temporal_axis="event_time",
        temporal_now=NOW,
        prefilter="never",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["created", "event", "modified"]
    assert diagnostics["temporal_axis_routed"] == "event_time"
