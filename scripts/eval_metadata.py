#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from leann.api import LeannSearcher, SearchResult
from leann.metadata_filter import MetadataFilterEngine

OUT_DIR = Path("/Users/zain/Documents/leann-eval-data/metadata-aware")
SPARSE_INDEX = OUT_DIR / "eval-sparse-corpus"
SLACK_INDEX = OUT_DIR / "eval-slack"
GOLD_PATH = Path("tests/eval/metadata_gold.jsonl")


def _load_gold() -> list[dict[str, Any]]:
    rows = []
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _result_satisfies_filters(result: SearchResult, metadata_filters: dict[str, Any]) -> bool:
    engine = MetadataFilterEngine()
    return bool(
        engine.apply_filters(
            [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.text,
                    "metadata": result.metadata,
                }
            ],
            metadata_filters,
        )
    )


def _group_cap_ok(
    results: list[SearchResult], diversify_by: str | list[str] | None, max_per_group: int | None
) -> bool:
    if diversify_by is None or max_per_group is None:
        return True
    fields = [diversify_by] if isinstance(diversify_by, str) else list(diversify_by)
    counts = Counter(tuple(result.metadata.get(field) for field in fields) for result in results)
    return all(count <= max_per_group for count in counts.values())


def _run_index(index_path: Path, gold: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not Path(f"{index_path}.meta.json").exists():
        raise FileNotFoundError(
            f"Missing eval index: {index_path}. Run build_metadata_eval_corpus.py"
        )

    false_zero_count = 0
    expected_positive_count = 0
    filter_checked = 0
    filter_ok = 0
    group_checked = 0
    group_ok = 0

    with LeannSearcher(
        str(index_path), enable_warmup=False, recompute_embeddings=False, use_daemon=False
    ) as searcher:
        for row in gold:
            metadata_filters = row.get("metadata_filters") or {}
            expected_min = int(row.get("expected_min_results", 1))
            if mode == "baseline":
                search_kwargs = {
                    "prefilter": "never",
                    "diversify_by": None,
                    "context_window": 0,
                }
            else:
                search_kwargs = {
                    "prefilter": row.get("prefilter", "auto"),
                    "diversify_by": row.get("diversify_by"),
                    "max_per_group": row.get("max_per_group", 2),
                    "context_window": row.get("context_window", 0),
                }

            results = searcher.search(
                row["query"],
                top_k=5,
                metadata_filters=metadata_filters,
                **search_kwargs,
            )

            if expected_min >= 1:
                expected_positive_count += 1
                if len(results) == 0:
                    false_zero_count += 1

            if results:
                filter_checked += len(results)
                filter_ok += sum(
                    1 for result in results if _result_satisfies_filters(result, metadata_filters)
                )

            expected_max_per_group = row.get("expected_max_per_group")
            if expected_max_per_group is not None and mode == "treatment":
                group_checked += 1
                if _group_cap_ok(results, row.get("diversify_by"), int(expected_max_per_group)):
                    group_ok += 1

    false_zero_rate = false_zero_count / expected_positive_count if expected_positive_count else 0.0
    filter_correctness = filter_ok / filter_checked if filter_checked else 1.0
    group_cap_compliance = group_ok / group_checked if group_checked else 1.0
    return {
        "mode": mode,
        "index": index_path.name,
        "queries": len(gold),
        "false_zero_rate": false_zero_rate,
        "group_cap_compliance": group_cap_compliance,
        "filter_correctness": filter_correctness,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("| mode | index | queries | false_zero_rate | group_cap | filter_correct |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['mode']} | {row['index']} | {row['queries']} | "
            f"{row['false_zero_rate']:.2%} | {row['group_cap_compliance']:.2%} | "
            f"{row['filter_correctness']:.2%} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--treatment", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_API_KEY", "local-iq")
    gold = _load_gold()
    mode_name = "baseline" if args.baseline else "treatment"
    rows = [_run_index(index_path, gold, mode_name) for index_path in (SPARSE_INDEX, SLACK_INDEX)]
    _print_table(rows)

    if args.treatment:
        failed = [
            row
            for row in rows
            if row["false_zero_rate"] >= 0.05
            or row["group_cap_compliance"] < 1.0
            or row["filter_correctness"] < 1.0
        ]
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
