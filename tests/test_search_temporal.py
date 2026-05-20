from datetime import datetime
from pathlib import Path

import pytest
from leann import LeannSearcher
from leann.metadata_filter import TEMPORAL_FALLBACK_FILTER

ROOT = Path(__file__).resolve().parents[1]
SLACK_INDEX = ROOT / ".leann" / "indexes" / "eval-slack" / "documents.leann"


def _require_slack_searcher():
    if not SLACK_INDEX.with_suffix(".leann.meta.json").exists():
        pytest.skip("Atom 4 eval-slack index has not been built")
    return LeannSearcher(str(SLACK_INDEX), recompute_embeddings=False, enable_warmup=False)


def _event_time(result):
    return datetime.fromisoformat(result.metadata["event_time"])


def test_search_without_temporal_returns_results_across_dates():
    searcher = _require_slack_searcher()

    results = searcher.search("Slack discussions", top_k=5, enable_temporal=False)

    assert len(results) == 5
    assert len({result.metadata["event_time"][:10] for result in results}) > 1


def test_search_with_temporal_filters_last_month_results():
    searcher = _require_slack_searcher()

    baseline = searcher.search("Slack discussions", top_k=5, enable_temporal=False)
    results = searcher.search("Slack discussions last month", top_k=5, enable_temporal=True)

    assert 0 < len(results) <= len(baseline)
    assert all(
        datetime(2026, 4, 1, tzinfo=_event_time(result).tzinfo)
        <= _event_time(result)
        <= datetime(2026, 4, 30, 23, 59, 59, 999999, tzinfo=_event_time(result).tzinfo)
        for result in results
    )


def test_search_temporal_overscan_trims_back_to_caller_top_k():
    """Regression: temporal_overscan multiplies the ANN candidate pool but must
    not leak past the caller's requested top_k. Was returning ~top_k * overscan
    before fix (api.py search() final trim in _return_with_diagnostics)."""
    searcher = _require_slack_searcher()

    results = searcher.search(
        "Slack discussions last month",
        top_k=3,
        enable_temporal=True,
        temporal_overscan=10,
    )
    assert len(results) <= 3

    results_strict = searcher.search(
        "Slack discussions last month",
        top_k=3,
        enable_temporal=True,
        temporal_strict=True,
        temporal_overscan=10,
    )
    assert len(results_strict) <= 3


def test_search_merges_caller_filters_with_parsed_temporal_filters(monkeypatch):
    searcher = _require_slack_searcher()
    captured_filters = []
    original_filter = searcher.passage_manager.filter_search_results

    def capture_filter(search_results, metadata_filters):
        captured_filters.append(metadata_filters)
        return original_filter(search_results, metadata_filters)

    monkeypatch.setattr(searcher.passage_manager, "filter_search_results", capture_filter)

    searcher.search(
        "Slack discussions last month",
        top_k=5,
        enable_temporal=True,
        metadata_filters={"source_type": {"==": "slack"}},
    )

    assert captured_filters
    assert captured_filters[-1][TEMPORAL_FALLBACK_FILTER]["axis"] == "event_time"
    assert captured_filters[-1]["source_type"] == {"==": "slack"}


def test_search_embeds_stripped_temporal_query(monkeypatch):
    searcher = _require_slack_searcher()
    embedded_queries = []
    original_compute = searcher.backend_impl.compute_query_embedding

    def capture_query(query, *args, **kwargs):
        embedded_queries.append(query)
        return original_compute(query, *args, **kwargs)

    monkeypatch.setattr(searcher.backend_impl, "compute_query_embedding", capture_query)

    searcher.search("Slack discussions last month", top_k=5, enable_temporal=True)

    assert embedded_queries[-1] == "Slack discussions"
