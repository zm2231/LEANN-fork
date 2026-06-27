from leann.api import SearchResult
from test_prefilter_integration import _searcher


def test_explain_filters_sparse_prefilter_reports_bruteforce_mode():
    searcher = _searcher(selectivity=0.03)

    results, diagnostics = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="auto",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["0", "1", "2"]
    assert diagnostics["total_passages"] == 100
    assert diagnostics["filter_matches"] == 3
    assert diagnostics["filter_selectivity"] == 0.03
    assert diagnostics["prefilter_mode_used"] == "bruteforce_filtered_subset"
    assert diagnostics["ann_candidates_requested"] == 0
    assert diagnostics["ann_candidates_returned"] == 0
    assert diagnostics["postfilter_survivors"] == 3
    assert diagnostics["results_returned"] == 3


def test_explain_filters_dense_filter_reports_ann_postfilter_mode():
    searcher = _searcher(selectivity=0.60)

    results, diagnostics = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "common"}},
        prefilter="auto",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["ann-0", "ann-1"]
    assert diagnostics["filter_matches"] == 60
    assert diagnostics["filter_selectivity"] == 0.60
    assert diagnostics["prefilter_mode_used"] == "ann_postfilter"
    assert diagnostics["ann_candidates_requested"] == 5
    assert diagnostics["ann_candidates_returned"] == 2
    assert diagnostics["postfilter_survivors"] == 2
    assert diagnostics["results_returned"] == 2


def test_explain_filters_flat_auto_reports_bruteforce_mode_for_dense_filter():
    searcher = _searcher(selectivity=0.60)
    searcher.backend_name = "flat"

    results, diagnostics = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="auto",
        explain_filters=True,
    )

    assert [result.id for result in results] == ["0", "1", "2"]
    assert diagnostics["filter_matches"] == 60
    assert diagnostics["filter_selectivity"] == 0.60
    assert diagnostics["prefilter_mode_used"] == "bruteforce_filtered_subset"
    assert diagnostics["ann_candidates_requested"] == 0
    assert diagnostics["ann_candidates_returned"] == 0
    assert diagnostics["postfilter_survivors"] == 3
    assert diagnostics["results_returned"] == 3


def test_explain_filters_no_filter_reports_no_filter_mode():
    searcher = _searcher(selectivity=0.03)

    results, diagnostics = searcher.search("query", top_k=5, explain_filters=True)

    assert [result.id for result in results] == ["ann-0", "ann-1"]
    assert diagnostics["total_passages"] == 100
    assert diagnostics["filter_matches"] == 100
    assert diagnostics["filter_selectivity"] == 1.0
    assert diagnostics["prefilter_mode_used"] == "no_filter"
    assert diagnostics["ann_candidates_requested"] == 5
    assert diagnostics["ann_candidates_returned"] == 2
    assert diagnostics["postfilter_survivors"] == 2
    assert diagnostics["results_returned"] == 2


def test_explain_filters_false_returns_results_directly():
    searcher = _searcher(selectivity=0.03)

    results = searcher.search(
        "query",
        top_k=5,
        metadata_filters={"channel": {"==": "rare"}},
        prefilter="auto",
        explain_filters=False,
    )

    assert isinstance(results, list)
    assert all(isinstance(result, SearchResult) for result in results)
