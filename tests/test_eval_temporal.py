# ruff: noqa: E402,I001
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import eval_temporal
from leann.api import SearchResult


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            SearchResult(
                id="0",
                score=1.0,
                text="",
                metadata={"source_id": "docs/plan.md#chunk-0"},
            )
        ]


def test_context_eval_matches_gold_ids_from_source_id():
    row = {
        "query": "docs/plan.md created in May",
        "gold_ids": ["docs/plan.md#chunk-0", "docs/plan.md#chunk-1"],
        "expected_min_results": 1,
        "bucket": "non_adversarial",
    }
    results = [
        SearchResult(
            id="0",
            score=1.0,
            text="",
            metadata={"source_id": "docs/plan.md#chunk-0"},
        )
    ]

    evaluated = eval_temporal.evaluate_context_row(row, results)

    assert evaluated.precision_at_5 == 0.2
    assert evaluated.recall_at_5 == 1.0
    assert evaluated.mrr == 1.0
    assert evaluated.bucket == "non_adversarial"


def test_context_treatment_forwards_temporal_options():
    searcher = FakeSearcher()
    row = {
        "query": "docs/plan.md edited in May",
        "now": "2026-05-19T12:00:00+00:00",
    }

    eval_temporal.search_context(searcher, row, "treatment", 20)

    assert searcher.calls == [
        (
            "docs/plan.md edited in May",
            {
                "top_k": 20,
                "enable_temporal": True,
                "gemma": 0.7,
                "temporal_now": datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
            },
        )
    ]


def test_summary_tracks_non_adversarial_recall():
    rows = [
        eval_temporal.EvalRow("q1", 0.2, 1.0, 1.0, 1, 20, "non_adversarial"),
        eval_temporal.EvalRow("q2", 0.0, 0.0, 0.0, 0, 20, "adversarial"),
    ]

    summary = eval_temporal.summarize("context", "treatment", rows)

    assert summary.precision_at_5 == 0.1
    assert summary.recall_at_5 == 0.5
    assert summary.non_adversarial_recall_at_5 == 1.0
