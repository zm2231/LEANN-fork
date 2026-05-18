from leann.api import LeannSearcher, SearchResult


def _result(result_id, score, **metadata):
    return SearchResult(id=result_id, score=score, text=f"text {result_id}", metadata=metadata)


def _searcher():
    return LeannSearcher.__new__(LeannSearcher)


def test_diversify_by_single_field_caps_each_group():
    results = [
        _result("a0", 1.0, source_document_id="doc-a"),
        _result("a1", 0.9, source_document_id="doc-a"),
        _result("a2", 0.8, source_document_id="doc-a"),
        _result("a3", 0.7, source_document_id="doc-a"),
        _result("a4", 0.6, source_document_id="doc-a"),
        _result("b0", 0.5, source_document_id="doc-b"),
        _result("b1", 0.4, source_document_id="doc-b"),
        _result("b2", 0.3, source_document_id="doc-b"),
        _result("b3", 0.2, source_document_id="doc-b"),
        _result("b4", 0.1, source_document_id="doc-b"),
    ]

    diversified = _searcher()._diversify_results(
        results, diversify_by="source_document_id", max_per_group=2, top_k=4
    )

    assert [result.id for result in diversified] == ["a0", "a1", "b0", "b1"]


def test_diversify_by_multiple_fields_uses_composite_key():
    results = [
        _result("0", 1.0, source_document_id="doc-a", author="u1"),
        _result("1", 0.9, source_document_id="doc-a", author="u1"),
        _result("2", 0.8, source_document_id="doc-a", author="u2"),
        _result("3", 0.7, source_document_id="doc-a", author="u2"),
    ]

    diversified = _searcher()._diversify_results(
        results,
        diversify_by=["source_document_id", "author"],
        max_per_group=1,
        top_k=4,
    )

    assert [result.id for result in diversified] == ["0", "2"]


def test_diversify_none_leaves_results_unchanged():
    results = [
        _result("0", 1.0, source_document_id="doc-a"),
        _result("1", 0.9, source_document_id="doc-a"),
        _result("2", 0.8, source_document_id="doc-a"),
    ]

    assert _searcher()._diversify_results(
        results, diversify_by=None, max_per_group=1, top_k=2
    ) == results


def test_diversify_returns_fewer_than_top_k_when_pool_exhausted():
    results = [
        _result("0", 1.0, source_document_id="doc-a"),
        _result("1", 0.9, source_document_id="doc-a"),
        _result("2", 0.8, source_document_id="doc-a"),
    ]

    diversified = _searcher()._diversify_results(
        results, diversify_by="source_document_id", max_per_group=1, top_k=3
    )

    assert [result.id for result in diversified] == ["0"]
