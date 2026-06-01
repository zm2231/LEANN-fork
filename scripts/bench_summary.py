#!/usr/bin/env python3
"""Summarize bench_vs_vanilla.py output: median + p95 + recall per config."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    rows = []
    for p in paths:
        with p.open() as f:
            for line in f:
                rows.append(json.loads(line))

    # Group by (label, query, config); compute median+p95 over trials
    groups = defaultdict(list)
    for r in rows:
        key = (r["label"], r["query"], r["config"])
        groups[key].append(r)

    # Also collect "ground truth" per (query, filter) = ours_native_prefilter result_ids
    # so we can compute recall of vanilla's post-filter approach.
    truth = {}
    for (label, query, cfg), trials in groups.items():
        if label == "ours" and cfg == "ours_native_prefilter":
            truth[query] = set(trials[0]["result_ids"])

    # Print table
    print(f"{'label':6s} {'query':22s} {'config':40s} {'n':>3s} {'med_ms':>8s} {'p95_ms':>8s} {'recall':>7s}")
    print("-" * 105)
    last_q = None
    for key in sorted(groups.keys(), key=lambda k: (k[1], k[2], k[0])):
        label, query, cfg = key
        if last_q is not None and query != last_q:
            print()
        last_q = query
        trials = groups[key]
        lats = sorted(t["latency_ms"] for t in trials)
        n_results = trials[0]["n_results"]
        med = statistics.median(lats)
        p95 = lats[int(0.95 * (len(lats) - 1))]
        recall_str = "-"
        if query in truth and truth[query]:
            t = truth[query]
            hits = sum(1 for t_id in t if t_id in set(trials[0]["result_ids"]))
            recall_str = f"{hits}/{len(t)}"
        print(f"{label:6s} {query:22s} {cfg:40s} {n_results:>3d} {med:>8.1f} {p95:>8.1f} {recall_str:>7s}")


if __name__ == "__main__":
    main()
