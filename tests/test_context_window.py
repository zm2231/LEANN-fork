from test_prefilter_integration import _searcher

from leann.api import LeannSearcher, SearchResult


def test_context_window_attaches_previous_and_next_siblings():
    searcher = _searcher(selectivity=0.03)

    results = searcher.search("query", top_k=1, context_window=1)

    assert len(results) == 2
    assert [sibling.id for sibling in results[0].siblings] == ["doc-ann:0", "doc-ann:2"]
    assert [sibling.id for sibling in results[1].siblings] == ["doc-ann:1", "doc-ann:3"]


def test_context_window_first_chunk_returns_only_next_sibling():
    searcher = _searcher(selectivity=0.03)
    hit = SearchResult(
        id="hit",
        score=1.0,
        text="hit text",
        metadata={"source_document_id": "doc-a", "chunk_seq": 0},
    )

    expanded = searcher.expand_context(hit, before=1, after=1)

    assert [sibling.id for sibling in expanded.siblings] == ["doc-a:1"]


def test_context_window_missing_source_document_id_leaves_siblings_none():
    searcher = _searcher(selectivity=0.03)
    hit = SearchResult(id="hit", score=1.0, text="hit text", metadata={"channel": "rare"})

    expanded = searcher.expand_context(hit, before=1, after=1)

    assert expanded.siblings is None


def test_context_window_does_not_deduplicate_hits_from_same_document():
    searcher = _searcher(selectivity=0.03)
    hits = [
        SearchResult(
            id="hit-1",
            score=1.0,
            text="hit 1",
            metadata={"source_document_id": "doc-a", "chunk_seq": 1},
        ),
        SearchResult(
            id="hit-2",
            score=0.9,
            text="hit 2",
            metadata={"source_document_id": "doc-a", "chunk_seq": 2},
        ),
    ]

    expanded = searcher._expand_context_results(hits, context_window=1)

    assert len(expanded) == 2
    assert [sibling.id for sibling in expanded[0].siblings] == ["doc-a:0", "doc-a:2"]
    assert [sibling.id for sibling in expanded[1].siblings] == ["doc-a:1", "doc-a:3"]


def test_expand_context_can_fetch_only_previous_siblings():
    searcher = _searcher(selectivity=0.03)
    hit = SearchResult(
        id="hit",
        score=1.0,
        text="hit text",
        metadata={"source_document_id": "doc-a", "chunk_seq": 3},
    )

    expanded = searcher.expand_context(hit, before=2, after=0)

    assert [sibling.id for sibling in expanded.siblings] == ["doc-a:1", "doc-a:2"]


def test_context_window_zero_leaves_result_objects_unchanged():
    searcher = LeannSearcher.__new__(LeannSearcher)
    hits = [SearchResult(id="hit", score=1.0, text="hit text", metadata={})]

    assert searcher._expand_context_results(hits, context_window=0) is hits
