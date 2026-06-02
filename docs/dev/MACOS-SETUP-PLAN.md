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
| B | Recompute latency: measure → in-process MLX (B2) or parallel batches (B1) | M | medium (may redirect) |
| C | Faster local builds (in-proc MLX build mode + parallel tokenize) | S | low |
| D | Prebuilt Apple-Silicon wheels (no source build) | L | medium (CI infra) |

**Key synergy:** B2 and C0 share one artifact — an **in-process MLX bge-m3 encoder**. Build it once (B2.1) and reuse for builds (C0). Sequence B before C0 so C0 consumes the same encoder.

**Sequencing:** A and D are independent and can run in parallel with B/C. B is measurement-gated (B0 may redirect B to B1 instead of B2). Recommended order: A (unblocks everyone) → B0 measurement → B1/B2 → C → D.

---

## Track A — Reproducible macOS bootstrap

| Atom | What | Done-test |
|---|---|---|
| A0 | `scripts/setup-macos.sh` (idempotent): verify `brew`, `brew install libomp boost protobuf zeromq pkgconf`, `git submodule update --init --recursive`, `uv sync`, smoke `uv run python -c "import leann"`. Non-zero exit + actionable message on each failure. Honors both `/opt/homebrew` (arm64) and `/usr/local` (Intel) prefixes. `Makefile` target `setup` wraps it. | Fresh clone on a Mac with none of the deps → one command → `uv run leann --help` works. |
| A1 | `leann doctor` subcommand in `cli.py`. Checks, each ✅/❌/⚠️ + fix command: Python ≥3.10; brew deps present (probe via `pkg-config --exists` for boost/zmq/protobuf, `brew --prefix libomp` for libomp); submodules initialized (`git submodule status` — leading `-` = missing); backend importable (`import leann_backend_hnsw`); MPS available (`torch.backends.mps.is_available()`); iq reachable (configurable endpoint, `GET /v1/models`, 3s timeout — ⚠️ not ❌, search can run `--no-recompute` without it). Exit 0 iff all *required* checks pass. | `leann doctor` on a healthy install exits 0, all ✅; with a submodule de-init'd, exits non-zero and names the fix. |
| A2 | Docs: rewrite the macOS section of `README.md` + `CLAUDE.md` build instructions to point at `make setup`; add a STATES.md note. Document the iq/embedding-endpoint prerequisite (see Gap G1). | README macOS quickstart is `make setup` + `leann doctor`; no manual brew/submodule steps remain. |
| A3 | Pre-flight guard: `leann build`/`search`/`ask` run a 2-check subset (backend importable +, when recompute is needed, iq reachable) and emit a one-line doctor-style error instead of a deep stack trace. | Point an index at a dead iq endpoint in recompute mode → friendly "embedding endpoint unreachable; run `leann doctor`" not a ZMQ traceback. |

---

## Track B — Recompute latency

**Premise to test, not assume.** The 30s is node-recompute. Earlier external analysis suggested the ~1.8s/batch is dominated by *embedding many long passages per HNSW step* (compute-bound), not HTTP RTT. If true, in-process MLX (B2) saves only the HTTP/serialization overhead — modest — and the real lever is **fewer/parallel recomputes** (B1). B0 decides. Do not build B2 before B0.

| Atom | What | Done-test |
|---|---|---|
| B0 | Instrument `hnsw_embedding_server.py` to log per-batch: `n_nodes`, `fetch_text_ms`, `embed_ms`, `serialize_ms`, `total_ms`. Run a fixed query set on `eval-slack-recompute`. Produce a breakdown: of the 30s, how much is iq HTTP/network vs MLX compute vs text fetch vs ZMQ serialization. | A table (atom commit in EXPERIMENTS.md) attributing the 30s across stages, with batch count and per-stage medians. |
| B1 | If HTTP/queueing is a material slice: parallelize the per-beam-step ZMQ→iq batches (concurrent requests to iq, bounded pool) and/or raise embedding-server concurrency. Keep recall identical. | A/B vs baseline: median query latency drops by the HTTP-slice fraction B0 predicted; recall@10 unchanged on the eval set. |
| B2 | If compute-bound or HTTP-bound, build an **in-process MLX bge-m3 encoder** (`embedding_compute.py` new provider, gated `embedding_mode="mlx-inproc"` / `LEANN_INPROC_MLX=1`): encode node batches directly via MLX in the server process, no HTTP hop. Must produce vectors numerically equivalent to iq's (same model). | Unit: in-proc MLX encode of N texts == iq encode within tolerance (cosine ≥ 0.999). Bench: query latency + recall vs HTTP path on `eval-slack-recompute`. |
| B3 | Decision + write-up: pick B1, B2, or both; record the measured wins and the rejected paths with numbers in EXPERIMENTS.md. Promote the winner from flag-gated to documented default for the macOS/iq stack if it clears a bar (target: <5s/query at unchanged recall). | EXPERIMENTS.md entry with before/after latency + recall; STATES.md updated with the chosen recompute path. |

---

## Track C — Faster local builds

| Atom | What | Done-test |
|---|---|---|
| C0 | Reuse B2's in-process MLX encoder as a first-class **build** embedding mode (builds currently go through iq HTTP "openai" mode or `sentence-transformers`/MPS). Lets index builds encode locally via MLX without the HTTP server. | Build a small index with `--embedding-mode mlx-inproc`; vectors match an iq-built index within tolerance; build wall-clock recorded vs the iq-HTTP build. |
| C1 | Port #209's parallel `tiktoken` truncation into `truncate_to_token_limit` (`embedding_compute.py`) — `ThreadPoolExecutor`, gated at >50 texts, GIL released by tiktoken. (Confirmed our current impl is a serial for-loop.) | Bench `truncate_to_token_limit` on ≥5k long texts: wall-clock drops on multi-core; output identical to serial for a fixed input. |

---

## Track D — Prebuilt Apple-Silicon wheels

| Atom | What | Done-test |
|---|---|---|
| D0 | `cibuildwheel` (or a manual CI matrix) job building `leann-backend-hnsw` for `macos-14`/arm64, Python 3.10–3.13: install brew deps + init submodules in the runner, compile, produce a wheel, `pip install` it in a clean venv and import. | CI artifact: an arm64 `leann_backend_hnsw-*.whl` that imports in a fresh venv with no C++/submodule present. |
| D1 | Host wheels (GitHub Releases on `zm2231/LEANN-fork`, or a private index) and document `uv pip install`/`--find-links` so a new Mac installs the wheel instead of building. | On a clean Mac with no brew C++ deps/submodules, install-from-wheel → `import leann_backend_hnsw` works. |
| D2 | Optional: wire `pyproject`/extras so `uv sync` prefers the published wheel when present, falling back to source build. | `uv sync` on a fresh clone pulls the wheel (no compile) when available; source build still works when not. |

Open feasibility questions for review: codesigning/notarization (likely unnecessary for local/tailnet use — confirm), wheel size/host, whether the faiss-vendored build is reproducible enough in CI, and whether D's payoff beats A's reliable source-build for a 1–3 Mac fleet.

---

## Gaps surfaced (beyond the four tracks)

- **G1 — iq/embedding-endpoint setup is undocumented.** Track A makes *LEANN* reproducible, but the embedding server (iq/MLX bge-m3) it depends on is a separate prerequisite with no setup doc. A truly reproducible Mac needs both. At minimum: document how iq is started + the env/flags LEANN needs; ideally a `leann doctor` check (A1 covers reachability) plus a runbook.
- **G2 — Endpoint config is hardcoded.** The skills and examples hardcode `100.122.112.83:8100`. If that host is offline the whole stack breaks. Consider a single source of truth (env/config) + `leann doctor` surfacing it. Touches A.
- **G3 — Test/extras profile.** Full `pytest` fails on `ivf`/`diskann` "backend not registered" because those extras aren't installed. A documented `make test` that installs the needed extras (or skips cleanly) would stop masking real failures. Small, fits A.
- **G4 — ZMQ embed cache (`LEANN_ZMQ_EMBED_CACHE`).** Already exists and defaults on; B0 should report its hit-rate, since a cheap recompute win may already be partially in place (or mis-tuned).

## Out of scope (later)
- Multi-index roll-up (Wave 6) and MCP `metadata_filters` passthrough — separate waves; no open upstream PR helps either (#199/#279 are stale/CONFLICTING).
- Linux/Windows setup parity — this wave is macOS-first.
- Content-hash passage IDs (#347) — IVF-unsafe `migrate-ids`; deferred.
