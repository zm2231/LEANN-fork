#!/usr/bin/env python3
"""Run Wave 1 temporal retrieval evals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from leann.api import LeannSearcher, SearchResult

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "tests" / "eval" / "temporal_gold.jsonl"
INDEXES = {
    "document": "eval-docs",
    "git_commit": "eval-commits",
    "slack": "eval-slack",
    "daily_summary": "eval-summaries",
}


@dataclass
class EvalRow:
    query: str
    precision_at_5: float
    recall_at_5: float
    mrr: float
    relevant_at_5: int
    returned: int


def load_gold() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with GOLD_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def in_expected_time(result: SearchResult, row: dict[str, Any]) -> bool:
    expected_range = row.get("expected_event_time_range")
    if not expected_range:
        return True
    event_time = result.metadata.get("event_time")
    if not event_time:
        return False
    parsed = parse_dt(event_time)
    return parse_dt(expected_range[0]) <= parsed <= parse_dt(expected_range[1])


def text_matches(result: SearchResult, row: dict[str, Any]) -> bool:
    needles = [needle.lower() for needle in row.get("expected_text_contains", [])]
    if not needles:
        return True
    haystack = f"{result.text} {json.dumps(result.metadata, sort_keys=True)}".lower()
    return any(needle in haystack for needle in needles)


def is_relevant(result: SearchResult, row: dict[str, Any]) -> bool:
    source_type = result.metadata.get("source_type")
    return (
        source_type in set(row["expected_source_types"])
        and in_expected_time(result, row)
        and text_matches(result, row)
    )


def open_searchers() -> dict[str, LeannSearcher]:
    searchers: dict[str, LeannSearcher] = {}
    for source_type, index_name in INDEXES.items():
        index_path = ROOT / ".leann" / "indexes" / index_name / "documents.leann"
        if not index_path.with_suffix(".leann.meta.json").exists():
            raise FileNotFoundError(f"Missing eval index for {source_type}: {index_path}")
        searchers[source_type] = LeannSearcher(str(index_path), recompute_embeddings=False)
    return searchers


def search_all(
    searchers: dict[str, LeannSearcher],
    query: str,
    mode: str,
    top_k_per_index: int,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    for searcher in searchers.values():
        kwargs: dict[str, Any] = {"top_k": top_k_per_index}
        if mode == "treatment":
            kwargs["enable_temporal"] = True
        merged.extend(searcher.search(query, **kwargs))
    return sorted(merged, key=lambda result: result.score, reverse=True)


def evaluate_row(
    row: dict[str, Any],
    results: list[SearchResult],
) -> EvalRow:
    top5 = results[:5]
    relevant_flags = [is_relevant(result, row) for result in top5]
    relevant_at_5 = sum(relevant_flags)
    expected_min = max(int(row.get("expected_min_results", 1)), 1)
    first_rank = next((idx + 1 for idx, flag in enumerate(relevant_flags) if flag), None)
    return EvalRow(
        query=row["query"],
        precision_at_5=relevant_at_5 / 5,
        recall_at_5=min(relevant_at_5 / expected_min, 1.0),
        mrr=0.0 if first_rank is None else 1.0 / first_rank,
        relevant_at_5=relevant_at_5,
        returned=len(results),
    )


def print_table(rows: list[EvalRow]) -> None:
    print("| query | precision@5 | recall@5 | mrr | relevant@5 | returned |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.query} | {row.precision_at_5:.2f} | {row.recall_at_5:.2f} | "
            f"{row.mrr:.2f} | {row.relevant_at_5} | {row.returned} |"
        )
    count = max(len(rows), 1)
    precision = sum(row.precision_at_5 for row in rows) / count
    recall = sum(row.recall_at_5 for row in rows) / count
    mrr = sum(row.mrr for row in rows) / count
    print(f"| AGGREGATE | {precision:.2f} | {recall:.2f} | {mrr:.2f} | - | - |")


def sanity_check(gold: list[dict[str, Any]], results_by_row: list[list[SearchResult]]) -> None:
    covered: set[str] = set()
    for gold_row, results in zip(gold, results_by_row):
        relevant = sum(is_relevant(result, gold_row) for result in results)
        if relevant >= int(gold_row.get("expected_min_results", 1)):
            covered.update(gold_row["expected_source_types"])
    missing = set(INDEXES) - covered
    if missing:
        raise SystemExit(
            f"Sanity check failed; no passing query for source types: {sorted(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true", help="Run raw-query baseline")
    mode.add_argument("--treatment", action="store_true", help="Run temporal treatment")
    parser.add_argument("--top-k-per-index", type=int, default=20)
    args = parser.parse_args()

    run_mode = "treatment" if args.treatment else "baseline"
    gold = load_gold()
    searchers = open_searchers()
    try:
        results_by_row = [
            search_all(searchers, row["query"], run_mode, args.top_k_per_index) for row in gold
        ]
        eval_rows = [evaluate_row(row, results) for row, results in zip(gold, results_by_row)]
    finally:
        for searcher in searchers.values():
            searcher.cleanup()
    print_table(eval_rows)
    sanity_check(gold, results_by_row)


if __name__ == "__main__":
    main()
