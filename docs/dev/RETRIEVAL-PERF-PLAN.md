# RETRIEVAL-PERF-PLAN — embedding round-trip cost & search batching

Brief for the recompute-perf codex wave. Consolidates what's already known so the wave starts from the frontier, not from scratch. Glossary in `STATES.md`; measurements in `EXPERIMENTS.md`.

## What's already settled (don't rediscover)

- **Upstream Issue #159 already diagnosed the 30s recompute search.** Conclusion (`ISSUE_159_CONCLUSION.md` on the upstream `issue-159-performance-analysis` branch, by maintainer andylizf): the cause is the default `complexity=64`; dropping to 16–32 took *their* setup from 36.17s → ~2.2–2.5s. `beam_width` is a DiskANN knob, irrelevant to HNSW. Our own sweep matches the direction (64→33s, 32→21s, 16→8.7s on eval-slack).
- **But complexity is necessary, not sufficient, for *our* topology.** Their 2.5s vs our 21s at complexity=32 (a ~10× gap) is because they embed on a 4090 GPU (~ms/batch) while we embed via MLX bge-m3 over iq HTTP (~120ms/batch localhost, +~100ms over Tailscale from the split-box deployment). complexity cuts the *number* of sequential beam steps; it does nothing about the *per-step embedding round-trip cost*, which is our real bottleneck.
- **The 30s is overhead-bound, not compute-bound.** ~60–120 sequentially dependent beam steps; per step = ZMQ REQ/REP (fresh socket each, `HNSW_zmq.cpp:545-563`) → `hnsw_embedding_server.py` → one iq HTTP call. Cross-step parallelism is impossible (step N+1 depends on N's distances). `LEANN_ZMQ_EMBED_CACHE` is dead code for search.
- **Storage is not a constraint here** (external drive). So `--no-recompute` (~55ms, larger files) is the correct *default*; recompute-speedup is about *reclaiming the option* at extreme scale, not urgent.

## Two workloads, both helped by batching

1. **`--no-recompute` (the actual default workload).** Each `search()` = ONE iq call (query embedding) + fast stored-vector ANN. For bulk/eval runs (many queries), the per-call ~35ms overhead × N dominates. **Batching N query embeddings into one iq call** is a direct win and needs no recompute at all.
2. **`--recompute` (the storage-saving option).** Per-query node round-trips dominate. Concurrent queries are *independent* (unlike beam steps), so their node-embedding requests can be **dynamically batched at the embedding server**.

## Levers, ranked for our topology

| Lever | Helps | Impact | Effort | Who owns |
|---|---|---|---|---|
| **A. complexity 16–32 (recompute default)** | recompute | 36s→~21s (ours), validated | S (one-liner) | LEANN |
| **B. `multi_search([q…])` — batch query embeddings** | **no-recompute bulk** + recompute | amortizes ~35ms/call across N queries; big for eval/bulk | S–M | LEANN |
| **C. Dynamic batching at the embedding server** | recompute throughput | merges concurrent queries' node requests into shared iq batches | M | iq (yours) + LEANN daemon |
| **D. Hot-node cross-query cache** (codex wave) | recompute | 2–20× (entry/hub nodes reused every query) | L (C++, half-built at `HNSW_zmq.cpp:201`) | LEANN/faiss |
| **E. iq binary/base64 response** | recompute + build | cuts ~100ms/batch network (350KB JSON → 64KB); iq currently ignores `encoding_format` | S–M | iq (yours) + LEANN decode |
| **F. In-process MLX encoder** | recompute + build | removes ~35ms HTTP overhead/call; network+MLX remain in split topology | M | LEANN/iq |

## The batching design (the new ask — B + C)

**B — `multi_search` (LEANN-side, no iq change):**
- New `LeannSearcher.multi_search(queries: list[str], top_k, **kw) -> list[list[SearchResult]]`.
- Embed all queries in ONE `compute_embeddings` call (the openai path already batches up to 500–800 inputs, `embedding_compute.py`), then run N ANN searches reusing the single embedding-server connection.
- For `--no-recompute` this is the whole win (1 iq call instead of N). For `--recompute`, it also lets the server see more nodes at once.
- Done-test: bench 50 queries on eval-slack `--no-recompute` as 50× `search()` vs 1× `multi_search()`; expect wall-clock to drop by ≈ (N−1)×per-call-overhead.

**C — dynamic batching at the embedding server (iq side):**
- Have iq (`mlx_embed_server.py`) collect requests arriving within a small window (e.g. 5–10ms), batch them into one MLX forward, and fan responses back. Standard inference-server pattern; you own iq.
- Pair with LEANN daemon mode (`use_daemon=True`, already present) so concurrent searches share one embedding server.
- Done-test: 8 concurrent recompute queries with vs without iq dynamic batching; expect throughput up ~Nx at similar per-query latency.

## What to tell the codex wave
- Quick wins A/B/E are mostly outside the wave (small, LEANN/iq edits).
- The wave's frontier is **D — the cross-query hot-node cache** (net-new; upstream's #159 fix was complexity + DiskANN + smaller model + no-recompute; nobody built the node cache). Seed: finish the top-degree disk-vector path (`HNSW_zmq.cpp:201-268`); `distances_batch` already splits disk vs remote nodes (`:624`).
- Our specific bottleneck is **embedding round-trip cost**, not complexity — so cache hit-rate on entry/upper-layer/hub nodes (reused by every query) is the thing to measure first.

## Out of scope
- DiskANN benchmark — deferred until a corpus genuinely outgrows RAM (at ~500MB the rubio/cadence indexes are RAM-resident; DiskANN's edge won't show).
