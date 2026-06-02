# MACOS-SETUP-PLAN — Wave 4 (macOS setup & embedding performance)

**Status:** draft for review. Branch when started: `feat/macos-setup` (off `feat/source-registry`).

**Why now (empirical):** standing up the fork on a second Mac (mini1, 2026-06-02) failed twice before working — `brew` was missing `libomp`, and the git clone never ran `git submodule update --init --recursive` so `packages/astchunk-leann` + the HNSW C++ deps (faiss, libzmq, cppzmq, msgpack-c) were empty and `uv sync` aborted. The fix was entirely manual. This wave codifies that, and folds in the macOS-native embedding-performance work that shares the same surface (MLX/iq).

**Glossary references:** see `docs/dev/STATES.md`. Terms used here:
- **iq** — the user's local embedding endpoint (MLX-served `BAAI/bge-m3`) at `http://100.122.112.83:8100/v1` (Tailscale) / `http://localhost:8100`. LEANN talks to it via the OpenAI-compatible embeddings API ("openai" embedding mode).
- **MPS** — Metal Performance Shaders; PyTorch's Apple-GPU backend, used by the `sentence-transformers` embedding path.
- **MLX** — Apple's array framework; what iq uses under the hood to serve bge-m3.
- **recompute** — LEANN's storage-saving mode: the index stores a pruned graph only; node embeddings are recomputed on demand during HNSW traversal via ZMQ → `hnsw_embedding_server.py` → iq HTTP. Measured ~30s/query on `eval-slack-recompute` (14,419 msgs), query-independent (node-recompute bound, not query-embed bound — verified 2026-06-02: identical-query repeat was no faster).
- **ZMQ embedding server** — `packages/leann-backend-hnsw/leann_backend_hnsw/hnsw_embedding_server.py`; receives node-id batches from the C++ HNSW core over ZMQ (port 5557), fetches passage text, embeds, returns vectors.

---

## Tracks (this wave bundles A + B + C + D)

| Track | Theme | Effort | Risk |
|---|---|---|---|
| A | Reproducible macOS bootstrap + `leann doctor` | S | low |
| B | Recompute latency: measure → default to `--no-recompute` for RAM-resident corpora; cheap wins; in-proc MLX only at scale | M | low (re-framed) |
| C | Faster local builds (in-proc MLX build mode; parallel-tokenize re-scoped or dropped) | S | low |
| D | **DEFERRED** — prebuilt arm64 wheels (non-relocatable without `delocate`; not worth L effort for 1–3 Macs) | L | high |

**Key synergy:** B3 (in-process MLX encoder) and C0 share one artifact. Build it once in B3, reuse for builds in C0. **C0 depends on "B3 built," not on all of B.**

**Sequencing (revised after review):** A is first and unblocks everyone. B is measurement-gated and re-prioritized: **B1 (make `--no-recompute` the default for RAM-resident corpora) is the real 30s→<1s win and is nearly free**; B2 (cheap recompute wins) is next; B3 (in-process MLX) is a *large-corpus-only* optimization, lowest priority. **D is deferred** (see Track D — non-relocatable wheel without `delocate`, not worth the L effort for a 1–3 Mac fleet). Recommended order: **A → B0 → B1 → B2 → (B3+C0 if a large corpus appears) → C1 only if justified.**

---

## Track A — Reproducible macOS bootstrap

| Atom | What | Done-test |
|---|---|---|
| A0 | `scripts/setup-macos.sh` (idempotent): verify `brew`, `brew install libomp protobuf zeromq pkgconf` (boost is **DiskANN-only** — `MSGPACK_NO_BOOST` in `CMakeLists.txt:53-55`, not an HNSW dep; install it only behind a `--diskann` flag), `git submodule update --init --recursive`, `uv sync`, smoke `uv run python -c "import leann"`. Non-zero exit + actionable message on each failure. Honors `/opt/homebrew` (arm64) and `/usr/local` (Intel) prefixes. `Makefile` target `setup` wraps it. | Fresh clone on a Mac with none of the deps → one command → `uv run leann --help` works. |
| A1 | `leann doctor` subcommand in `cli.py` (register parser near `cli.py:810`; add the `elif args.command == "doctor"` dispatch **before** the `_handle_plugin_command` fallthrough at `cli.py:3372`, which returns truthy and can swallow unknown commands). Checks, each ✅/❌/⚠️ + fix command: Python ≥3.10; brew deps present — **probe via `brew --prefix <formula>` for all five** (verified: `pkg-config --exists` works only for `libzmq` and `protobuf`; boost+libomp ship no `.pc` and false-negative; and the package name is `libzmq`, not `zmq`); submodules initialized (`git submodule status` — leading `-` = missing, verified reliable); backend importable (`import leann_backend_hnsw`); MPS available (`torch.backends.mps.is_available()`); iq reachable (endpoint from config/env, `GET /v1/models`, 3s timeout — ⚠️ not ❌, since `--no-recompute` search needs no server). Exit 0 iff all *required* checks pass. | `leann doctor` on a healthy install exits 0, all ✅; with a submodule de-init'd, exits non-zero and names the fix. |
| A2 | Docs: rewrite the macOS section of `README.md` + `CLAUDE.md` build instructions to point at `make setup`; add a STATES.md note. Document the iq/embedding-endpoint prerequisite (see Gap G1). | README macOS quickstart is `make setup` + `leann doctor`; no manual brew/submodule steps remain. |
| A3 | Pre-flight guard: `leann build`/`search`/`ask` run a 2-check subset (backend importable +, when recompute is needed, iq reachable) and emit a one-line doctor-style error instead of a deep stack trace. | Point an index at a dead iq endpoint in recompute mode → friendly "embedding endpoint unreachable; run `leann doctor`" not a ZMQ traceback. |

---

## Track B — Recompute latency (RE-FRAMED after codex review)

**Corrected premise.** The 30s is **per-round-trip overhead, not MLX compute.** Verified from the C++: search calls `distances_batch` → `fetch_distances_zmq` (`HNSW_search.cpp:776`, `HNSW_zmq.cpp:618`); each of ~60–120 *sequentially dependent* beam steps (`complexity=64, beam_width=1`) issues one ZMQ REQ/REP that **opens a fresh socket per batch** (`HNSW_zmq.cpp:545-563,613`) → one iq HTTP call. Slack passages are short (median 53 / p90 159 tokens, n=2000), so the MLX forward is single-digit ms; the ~300ms/batch is ZMQ RTT + socket churn + HTTP RTT + iq scheduling + msgpack. Consequences:
- **Cross-step parallelism is impossible** (step N+1's candidates depend on step N's distances, `HNSW_search.cpp:779-793`) and within-step is already one batched call → **the old "B1 parallelize batches" is dropped.**
- **`LEANN_ZMQ_EMBED_CACHE` is a no-op for search** — `distances_batch` never reads/writes `cached_vectors` (that map is only used by the unused `get_vector_zmq` path, `HNSW_zmq.cpp:473-510`). It is NOT a partial win; it is dead code for queries.
- **The dominant lever is to not recompute at all when the corpus is RAM-resident.** The 14,419×1024×4B ≈ 59 MB of vectors fits trivially; a `--no-recompute` (full-vector, non-pruned) HNSW does pure in-memory FAISS search (`api.py:626-635`) → **30s → <1s, zero embedding-server traffic.** The product `eval-slack` index is *already* `--no-recompute`; the 30s only appears in the recompute variant, which is unnecessary at this scale. In-process MLX matters **only** for corpora genuinely too large to store vectors.

| Atom | What | Done-test |
|---|---|---|
| B0 | Instrument `hnsw_embedding_server.py` per-batch (`n_nodes`, `fetch_text_ms`, `embed_ms`, `serialize_ms`, `total_ms`) + count beam steps. Run a fixed query set on `eval-slack-recompute`. Confirm/quantify the overhead-vs-compute split. | EXPERIMENTS.md table attributing the 30s across stages with batch count + per-stage medians. |
| B1 | **Make `--no-recompute` the explicit default path for RAM-resident corpora.** Add a build-time heuristic: when estimated stored-vector size (ntotal×dim×4B) is below a threshold (e.g. <2 GB), recommend/emit `--no-recompute` (or warn that recompute trades ~1000× latency for storage you don't need). Document in `leann build` help + STATES.md. (No format change; this is guidance + a warning, plus a `leann doctor`/build note.) | Building a small corpus with `--recompute` prints a one-line "corpus fits in RAM; `--no-recompute` is ~1000× faster here" notice; `--no-recompute` path unaffected. |
| B2 | **Cheap recompute wins (for when recompute IS needed).** (a) Reuse one persistent ZMQ socket across batches instead of connect/close per batch (`HNSW_zmq.cpp:545-563`). (b) Lower the default search `complexity` on the recompute path (64→32, matching `ask`, `cli.py:622`) → proportionally fewer beam steps/round-trips. (c) Wire a **persistent node-embedding cache** into `distances_batch` (sidecar file, reused across queries) so overlapping/repeat queries skip re-embedding — the half-built `HNSW_zmq.cpp:201-268` disk path is a starting point. | A/B on `eval-slack-recompute`: socket-reuse + complexity=32 cut median latency measurably at unchanged recall@10; second identical query is materially faster with the persistent cache (vs the measured no-speedup today). |
| B3 | **In-process MLX encoder — only for corpora too large for `--no-recompute`.** New `embedding_compute.py` provider gated `embedding_mode="mlx-inproc"` / `LEANN_INPROC_MLX=1`, encoding node batches directly via MLX in the server process (removes the HTTP hop + msgpack; candidate libs: `mlx_embedding_models`, `mlx-community/bge-m3-mlx-fp16`). Must match iq's vectors **including pooling** (iq's pooling vs our mean-pool, `embedding_compute.py:802`). | Unit: in-proc MLX encode == iq encode, cosine ≥ 0.999 on 100 texts. Bench: recompute query latency + recall vs HTTP path, on a corpus too big for `--no-recompute`. |
| B4 | Decision + write-up: record measured wins and rejected paths with numbers. Promote winners (B1 guidance, B2 cheap wins) to defaults; keep B3 flag-gated for the large-corpus case. | EXPERIMENTS.md before/after latency+recall; STATES.md updated with the recompute-path guidance. |

---

## Track C — Faster local builds

| Atom | What | Done-test |
|---|---|---|
| C0 | Reuse **B3's** in-process MLX encoder as a first-class **build** embedding mode (builds currently go through iq HTTP "openai" mode or `sentence-transformers`/MPS; build + recompute share one dispatch at `embedding_compute.py:356-396`, build path is `compute_embeddings(... is_build=True)` → `compute_embeddings_direct`, `api.py:78-90,786`, so one new `elif` serves both). Depends on **B3 built**, not all of B. Verify numeric equivalence incl. pooling. | Build a small index with `--embedding-mode mlx-inproc`; vectors match an iq-built index (cosine ≥ 0.999); build wall-clock recorded vs the iq-HTTP build. |
| C1 | **Re-scoped (was near-worthless as first written).** Measured: `truncate_to_token_limit` (`embedding_compute.py:134`, serial loop at :156) does 5,000 texts in 0.078s, and slack p90=159 tokens vs bge-m3's 8192 limit → it truncates ~nothing. Parallel `tiktoken` only helps a **large corpus of genuinely long documents (>8k tokens)**. Either (a) gate the `ThreadPoolExecutor` port to build-time + only when median text length is high, or (b) **drop C1**. Do NOT let it fire on recompute batches. | If kept: bench on a long-document corpus shows wall-clock drop; output identical to serial. If dropped: removed from plan with the measurement noted. |

---

## Track D — Prebuilt Apple-Silicon wheels — **DEFERRED**

**Why deferred (review finding):** the wheel is **not relocatable as built.** `CMakeLists.txt:21` hard-links `/opt/homebrew/opt/libomp/lib/libomp.dylib` and `:41` links `libzmq` via a pkg-config IMPORTED_TARGET, so the compiled `.so` carries install-name references to Homebrew dylibs. On any Mac lacking those exact paths, `import leann_backend_hnsw` fails with a dyld error. A portable wheel therefore **requires `delocate-wheel`** to vendor `libomp.dylib` + `libzmq.dylib` and rewrite install names — a step the original plan never named. Given that and the L-effort CI (scikit-build-core + CMake + SWIG + vendored faiss, `pyproject.toml:3-5`), Track A's reliable source build already solves the 1–3 Mac fleet problem at far lower cost. Codesigning/notarization is unnecessary for tailnet use.

**If revisited:** downgrade to a single S atom — document a *local* `uv build` + `delocate-wheel` recipe to produce one portable arm64 wheel on demand (no CI), and only graduate to `cibuildwheel` if the fleet grows. Do **not** ship D0/D1 as originally written — they produce a broken wheel.

---

## Gaps surfaced (beyond the four tracks)

- **G1 — iq/embedding-endpoint setup is undocumented.** Track A makes *LEANN* reproducible, but the embedding server (iq/MLX bge-m3) it depends on is a separate prerequisite with no setup doc. A truly reproducible Mac needs both. At minimum: document how iq is started + the env/flags LEANN needs; ideally a `leann doctor` check (A1 covers reachability) plus a runbook.
- **G2 — Endpoint config is hardcoded.** The skills and examples hardcode `100.122.112.83:8100`. If that host is offline the whole stack breaks. Consider a single source of truth (env/config) + `leann doctor` surfacing it. Touches A.
- **G3 — Test/extras profile.** Full `pytest` fails on `ivf`/`diskann` "backend not registered" because those extras aren't installed. A documented `make test` that installs the needed extras (or skips cleanly) would stop masking real failures. Small, fits A.
- **G2 fix detail (validated):** `embedding_options.base_url` is baked into `.meta.json` at build time (the `100.122.112.83:8100` literal), so an index is pinned to one endpoint. Fix = a `LEANN_EMBEDDING_BASE_URL` override resolved at search time, surfaced by `leann doctor`. Folds into A.
- **G4 — CORRECTED: `LEANN_ZMQ_EMBED_CACHE` is dead code for search, not a partial win.** The search hot path (`distances_batch`/`fetch_distances_zmq`) never touches `cached_vectors`; that map lives only in the unused `get_vector_zmq` path (`HNSW_zmq.cpp:473-510`), and `reset_fetch_count` clears state every query (`HNSW_zmq.h:134-141`). So the env flag does nothing for query latency. Either stop advertising it as a perf knob, **or** wire a persistent node-embedding cache into `distances_batch` — that wiring is exactly atom **B2(c)** and is genuinely high-value (cross-query reuse).

## Out of scope (later)
- Multi-index roll-up (Wave 6) and MCP `metadata_filters` passthrough — separate waves; no open upstream PR helps either (#199/#279 are stale/CONFLICTING).
- Linux/Windows setup parity — this wave is macOS-first.
- Content-hash passage IDs (#347) — IVF-unsafe `migrate-ids`; deferred.
