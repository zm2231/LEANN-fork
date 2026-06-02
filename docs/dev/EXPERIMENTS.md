# EXPERIMENTS — benchmarks & A/B results

Glossary in `docs/dev/STATES.md`. All runs on eval-slack (14,419 Slack messages). "iq" = local MLX bge-m3 embedding server (`forks/iq/mlx_embed_server.py`, port 8100) on the dedicated-embedding Mac. mini1 = the Mac where LEANN search/build runs, reaching iq over Tailscale.

## 2026-06-02: BM25/FTS5 hybrid fusion — bug + fix

After the upstream sync replaced the in-memory `BM25Scorer` with `Fts5BM25Index` (SQLite FTS5), hybrid search (`search(vector_weight=...)`, where 1.0=pure vector, 0.0=pure BM25) was benchmarked for the first time on eval-slack (`scripts/bench_hybrid.py`, top-10).

**Symptom (before fix):** `overlap@vec` = fraction of the pure-vector (weight 1.0) top-10 retained at a given weight.

| vector_weight | warm latency | overlap@vec (BEFORE) | overlap@vec (AFTER) |
|---|---|---|---|
| 1.0 (pure vector) | 56 ms | 1.00 | 1.00 |
| 0.9 | 56 ms | — | 0.77 |
| 0.7 | 63 ms | **0.06** | 0.58 |
| 0.5 | 63 ms | **0.06** | 0.47 |
| 0.3 | 55 ms | — | 0.34 |
| 0.0 (pure BM25) | **0.6 ms** | 0.06 | 0.06 |

Before the fix, weights 0.7/0.5 returned the SAME results as pure BM25 (overlap 0.06) — `vector_weight` was effectively a no-op; BM25 dominated at every weight.

**Root cause:** raw linear fusion `vector_weight*vec_score + (1-vector_weight)*bm25_score` (api.py hybrid block) summed incomparable scales — vector cosine/IP ≈ [0,1] vs FTS5 `bm25()` (negated to higher-is-better) unbounded, often 2–15. At vector_weight=0.7 the vector term ≈ 0.7×0.68 ≈ 0.48 while the BM25 term ≈ 0.3×(2..15) easily exceeded it, so BM25 won outright.

**Fix (commit cd15cd7):** min-max normalize each source to [0,1] before the weighted sum; a doc absent from one source contributes 0 for it. After the fix, `overlap@vec` is a clean monotonic gradient (1.0 → 0.77 → 0.58 → 0.47 → 0.34 → 0.06), i.e. `vector_weight` is an honest linear blend. Test assertion relaxed `> 0` → `>= 0` (min-max floors the worst candidate to exactly 0).

**FTS5 itself:** excellent — pure BM25 is **0.6 ms** (SQLite indexed lookup), build cost negligible at 14k passages. The migration's mechanical win is real.

## 2026-06-02: Recompute latency — cost decomposition (real split-box topology)

The dedicated-embedding Mac runs iq; mini1 runs LEANN search and calls iq over Tailscale. A recompute query = ~60–120 sequentially dependent HNSW beam steps, each one batched ZMQ → `hnsw_embedding_server.py` → one iq HTTP `/v1/embeddings` call.

**iq `/v1/embeddings` latency, localhost (median of 5):**

| batch | latency | interpretation |
|---|---|---|
| 1 | 38 ms | ≈ fixed per-call overhead (HTTP/SDK/queue) |
| 16 | 118 ms | → ~5 ms/text MLX forward |
| 64 | 304 ms | |

So per call ≈ **~35 ms fixed overhead + ~5 ms/text MLX forward**.

**Network (mini1 → iq over Tailscale, batch=16, 5×):** ~211–247 ms (vs 118 ms localhost) — **+~100 ms/call**. Tailscale *connect* is only ~11 ms; the +100 ms is the **response payload**: a batch-16 embedding response is **350 KB of JSON floats** vs **64 KB binary float32 / 32 KB fp16** (5–11× bloat). The fat JSON, not the network path, is the cost.

**Implication:** production recompute (mini1→iq) is ~35 s+, *worse* than the 30 s localhost measurement. Per real round trip (~221 ms): ~35 ms HTTP overhead + ~80 ms MLX + ~100 ms fat-JSON network. Levers, topology-aware: (1) hot-node cache on mini1 (avoid the round trip), (2) **binary/fp16 embedding response in iq** — user owns iq; kills most of the ~100 ms network slice; neither prior review caught this since both assumed localhost, (3) in-process MLX removes only the ~35 ms fixed overhead in this split topology, (4) fewer nodes (lower complexity).

## 2026-06-02: Recompute knob sweep (eval-slack-recompute, query "design feedback")

| Config | Time | Result overlap |
|---|---|---|
| complexity=64, beam_width=1, batch_size=0 (baseline) | 33.2 s | — |
| beam_width=8 | 40.0 s | 5/5 identical |
| batch_size=128 | 40.4 s | 5/5 identical |
| **complexity=32** | **21.1 s** | **5/5 identical** |
| complexity=16 | 8.7 s | 4/5 |

**beam_width and batch_size are anti-levers** (they widen per-step work → slower, no recall gain). **complexity=32 is the only clean win** (~1.6×, 5/5 recall). complexity=16 is fast (~3.8×) but recall-risky (4/5).

## Storage facts (eval-slack, 14,419 passages)

- `--no-recompute` dir: 73 MB; `--recompute` dir: 17 MB. Vector index drops 60 MB → 3.7 MB under recompute (the ~97% storage-reduction value prop). At extreme scale, `--no-recompute` is infeasible → recompute is mandatory.
- Chunk size default = 256 tokens (overlap 128); bge-m3 token limit = 8192 → **~32× headroom**. Chunk size is NOT memory-gated; at extreme scale it is a recompute/storage lever (bigger chunks = fewer graph nodes = fewer recompute round trips + less storage, trading retrieval granularity).
