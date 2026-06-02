#!/usr/bin/env python3
"""Benchmark BM25/FTS5 hybrid search vs pure-vector on a --no-recompute index.

Measures, per vector_weight: first-query latency (includes one-time FTS5 build
on first BM25 use), warm median/p95 latency, and result-set overlap vs pure
vector (vector_weight=1.0) as a quality-shift proxy.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from leann import LeannSearcher

QUERIES = [
    "discussions about task triage",
    "BTD landing page",
    "design feedback",
    "Loom video demos",
    "model evaluation benchmarks",
    "Apple submission process",
    "weekly summary",
    "prompt engineering tips",
    "API integration plan",
    "Stunspot prompt library",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--weights", default="1.0,0.7,0.5,0.0")
    args = ap.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    s = LeannSearcher(args.index, recompute_embeddings=False, enable_warmup=False)

    # Truth = pure vector (weight 1.0) result ids per query
    truth = {}
    rows = []
    for w in weights:
        first_latency = {}
        warm_latency = {}
        result_ids = {}
        for qi, q in enumerate(QUERIES):
            lats = []
            for t in range(args.trials):
                t0 = time.perf_counter()
                res = s.search(q, top_k=args.top_k, vector_weight=w)
                dt = (time.perf_counter() - t0) * 1000
                if t == 0:
                    first_latency[q] = dt
                else:
                    lats.append(dt)
                if t == args.trials - 1:
                    result_ids[q] = [r.id for r in res]
            warm_latency[q] = statistics.median(lats) if lats else first_latency[q]
        if w == 1.0:
            truth = dict(result_ids)
        # overlap vs truth
        overlaps = []
        for q in QUERIES:
            if q in truth and truth[q]:
                inter = len(set(result_ids[q]) & set(truth[q]))
                overlaps.append(inter / len(truth[q]))
        rows.append({
            "weight": w,
            "first_med_ms": statistics.median(first_latency.values()),
            "warm_med_ms": statistics.median(warm_latency.values()),
            "warm_p95_ms": sorted(warm_latency.values())[int(0.95 * (len(QUERIES) - 1))],
            "overlap_vs_vector": statistics.mean(overlaps) if overlaps else 1.0,
        })

    print(f"{'vector_weight':14s} {'first_med_ms':>13s} {'warm_med_ms':>12s} {'warm_p95_ms':>12s} {'overlap@vec':>11s}")
    print("-" * 66)
    for r in rows:
        print(f"{r['weight']:<14.2f} {r['first_med_ms']:>13.1f} {r['warm_med_ms']:>12.1f} "
              f"{r['warm_p95_ms']:>12.1f} {r['overlap_vs_vector']:>11.2f}")
    print("\n(overlap@vec = mean fraction of pure-vector top-k retained at this weight; "
          "1.0=identical, lower=more BM25 influence)")

    out = Path("docs/dev/bench-runs/hybrid.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
