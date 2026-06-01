#!/usr/bin/env python3
"""Bench LEANN search latency: vanilla vs fork.

Invoked twice with different interpreters:
  ~/.local/share/uv/tools/leann-core/bin/python scripts/bench_vs_vanilla.py --label vanilla ...
  .venv/bin/python scripts/bench_vs_vanilla.py --label ours ...

Vanilla path simulates what a user would do without metadata filter support:
  search(top_k=oversample), then drop non-matching results in Python.
Ours path: search(top_k=N, metadata_filters=...) — native prefilter.

Output: one JSON line per (config, query, trial) to --output.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from leann import LeannSearcher

QUERIES = [
    {"q": "discussions about task triage", "filter_channel": "channel:big-brain", "label": "big-brain"},
    {"q": "BTD landing page", "filter_channel": "channel:btd-community", "label": "btd"},
    {"q": "design feedback", "filter_channel": "channel:product-strategy", "label": "product"},
    {"q": "Stunspot prompt library", "filter_author": "U09KG04U8LS", "label": "author-stunspot"},
    {"q": "Loom video demos", "label": "no-filter-loom"},
    {"q": "model evaluation benchmarks", "label": "no-filter-evals"},
    {"q": "Apple submission process", "label": "no-filter-apple"},
    {"q": "weekly summary", "filter_channel": "channel:Tam", "label": "tam"},
    {"q": "prompt engineering tips", "filter_recent_month": "2026-05", "label": "recent-may"},
    {"q": "API integration plan", "filter_recent_month": "2026-04", "label": "recent-apr"},
]


def _matches(result, filt):
    md = result.metadata or {}
    if "channel" in filt:
        if md.get("parent_ref") != filt["channel"]:
            return False
    if "author" in filt:
        if md.get("author") != filt["author"]:
            return False
    if "month_prefix" in filt:
        ts = md.get("created_at", "")
        if not isinstance(ts, str) or not ts.startswith(filt["month_prefix"]):
            return False
    return True


def run_vanilla(searcher, query, filt, top_k, oversample):
    """Vanilla path: search wide, post-filter in Python."""
    t0 = time.perf_counter()
    raw = searcher.search(query, top_k=oversample, complexity=64)
    if filt:
        raw = [r for r in raw if _matches(r, filt)]
    raw = raw[:top_k]
    dt_ms = (time.perf_counter() - t0) * 1000
    return dt_ms, raw


def run_ours_filter(searcher, query, filt, top_k):
    """Ours path: native prefilter via metadata_filters."""
    metadata_filters = {}
    if "channel" in filt:
        metadata_filters["parent_ref"] = {"==": filt["channel"]}
    if "author" in filt:
        metadata_filters["author"] = {"==": filt["author"]}
    if "month_prefix" in filt:
        # created_at >= start of month, < start of next
        prefix = filt["month_prefix"]
        y, m = prefix.split("-")
        start = f"{y}-{m}-01T00:00:00+00:00"
        ny, nm = (int(y), int(m) + 1) if int(m) < 12 else (int(y) + 1, 1)
        end = f"{ny:04d}-{nm:02d}-01T00:00:00+00:00"
        metadata_filters["created_at"] = {">=": start, "<": end}
    t0 = time.perf_counter()
    res = searcher.search(
        query,
        top_k=top_k,
        complexity=64,
        metadata_filters=metadata_filters if metadata_filters else None,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    return dt_ms, res


def run_no_filter(searcher, query, top_k):
    t0 = time.perf_counter()
    res = searcher.search(query, top_k=top_k, complexity=64)
    dt_ms = (time.perf_counter() - t0) * 1000
    return dt_ms, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to .leann index (no extension)")
    ap.add_argument("--label", required=True, choices=["vanilla", "ours"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--recompute", action="store_true",
                    help="Pass to LeannSearcher when querying a --recompute index")
    ap.add_argument("--oversample-values", default="50,200,500",
                    help="Comma-sep oversample top_k values for vanilla post-filter")
    args = ap.parse_args()

    oversamples = [int(x) for x in args.oversample_values.split(",")]

    # Detect whether we're on the fork (has recompute autodetect / matching_filtered_subset)
    import leann.api as la
    has_autodetect = "recompute_embeddings" in la.LeannSearcher.__init__.__code__.co_varnames
    is_fork = hasattr(la, "_AUTODETECT_RECOMPUTE_SENTINEL") or hasattr(
        la.LeannSearcher, "_resolve_recompute"
    ) or "matching_filtered_subset" in open(la.__file__).read()

    print(f"[{args.label}] is_fork={is_fork} leann={la.__file__}", file=sys.stderr)

    # Ours: pass recompute=None to autodetect from meta (it should pick False).
    # Vanilla: must pass False explicitly (its default is True, which would trigger embed-recompute).
    searcher = LeannSearcher(
        args.index,
        enable_warmup=False,
        recompute_embeddings=args.recompute,
        use_daemon=False,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        # Warm up: run each unique query once and discard
        for q in QUERIES[:3]:
            _ = run_no_filter(searcher, q["q"], 5)

        for trial in range(args.trials):
            for q in QUERIES:
                base_filt = {}
                if "filter_channel" in q:
                    base_filt["channel"] = q["filter_channel"]
                if "filter_author" in q:
                    base_filt["author"] = q["filter_author"]
                if "filter_recent_month" in q:
                    base_filt["month_prefix"] = q["filter_recent_month"]

                # Always run no-filter for B1
                dt, res = run_no_filter(searcher, q["q"], args.top_k)
                f.write(json.dumps({
                    "label": args.label, "trial": trial, "query": q["label"],
                    "config": "no_filter", "filter": None, "top_k": args.top_k,
                    "latency_ms": dt, "n_results": len(res),
                    "result_ids": [r.id for r in res],
                }) + "\n")

                if not base_filt:
                    continue  # B1 only for no-filter queries

                if args.label == "vanilla":
                    for k in oversamples:
                        dt, res = run_vanilla(searcher, q["q"], base_filt, args.top_k, k)
                        f.write(json.dumps({
                            "label": args.label, "trial": trial, "query": q["label"],
                            "config": f"vanilla_postfilter_oversample={k}", "filter": base_filt,
                            "top_k": args.top_k, "oversample": k,
                            "latency_ms": dt, "n_results": len(res),
                            "result_ids": [r.id for r in res],
                        }) + "\n")
                else:
                    # ours: native prefilter, auto-routed
                    dt, res = run_ours_filter(searcher, q["q"], base_filt, args.top_k)
                    f.write(json.dumps({
                        "label": args.label, "trial": trial, "query": q["label"],
                        "config": "ours_native_prefilter", "filter": base_filt,
                        "top_k": args.top_k,
                        "latency_ms": dt, "n_results": len(res),
                        "result_ids": [r.id for r in res],
                    }) + "\n")
                    # Also run vanilla-style post-filter via ours for direct head-to-head
                    for k in oversamples:
                        dt, res = run_vanilla(searcher, q["q"], base_filt, args.top_k, k)
                        f.write(json.dumps({
                            "label": args.label, "trial": trial, "query": q["label"],
                            "config": f"ours_postfilter_oversample={k}", "filter": base_filt,
                            "top_k": args.top_k, "oversample": k,
                            "latency_ms": dt, "n_results": len(res),
                            "result_ids": [r.id for r in res],
                        }) + "\n")

    searcher.close() if hasattr(searcher, "close") else None
    print(f"[{args.label}] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
