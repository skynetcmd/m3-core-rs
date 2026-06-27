# m3-core-rs Benchmarks

Per-operation speed measurements for the crates in this workspace, as consumed
through the `m3-core-py` PyO3 bindings from m3-memory. The guiding principle
(shared with m3-memory's `docs/OXIDATION_BENCHMARKS.md`) is **honest,
per-operation reporting**: large wins are stated with the input size that
produces them, break-even and losses are stated too, and no single headline
multiplier is presented as "the" speedup.

For the established vector / redaction / FTS numbers (`mmr_rerank`,
`cosine_batch`, `redaction`, `sanitize_fts`, …) see m3-memory's
`docs/OXIDATION_BENCHMARKS.md`, which benches them FFI-inclusive against the
production Python paths. This document covers the **Milestone-4 additions**:
`m3-ingest` (filesystem walk + batch hashing) and `m3-governor`.

## Run context

| | |
|---|---|
| Date | 2026-06-27 |
| Crate version | m3-core-rs 3.6.22 (Windows CUDA wheel, cp314) |
| Python | 3.14.3 |
| Platform | Windows 11 (10.0.26200), AMD64, NVIDIA RTX 5080 |
| Timing | `time.perf_counter`; warm cache; median of repeated runs |
| Parity | every result asserts native output == Python reference before timing |

---

## `m3-ingest` — batch content hashing (`hash_files`)

The staleness/re-ingest sweep hashes every file whose mtime changed since the
last ingest. The Python path (`file_content_sha256`) is a serial per-file
streaming SHA-256 loop. `m3_ingest::hash_files` reads + hashes the whole batch
in parallel with `rayon`, releasing the GIL, reusing the FIPS-aware `m3-hash`
provider so digests are byte-identical.

Speedup = Python serial median ÷ native median. Parity verified (native digest
== `hashlib.sha256` per file) on every run.

| Input | Python serial | native `hash_files` | Speedup | Verdict |
|---|---:|---:|---:|---|
| 500 files × 64 KiB | 43.5 ms | 6.7 ms | **6.45×** | rust faster |
| 1000 files × 128 KiB | 117.1 ms | 16.8 ms | **6.96×** | rust faster |

**Honest reading.** The win is real and scales with batch size because the work
is embarrassingly parallel (independent file reads + hashes) and the per-call
FFI cost is amortized across the whole batch. This is the opposite regime from
single-file `sha256_hex` (which m3-memory benches as break-even-to-slower,
because `hashlib` is already C and a single small hash doesn't cover the FFI
crossing). The rule that falls out: **hash in batches through Rust, hash one
file through Python.** m3-memory wires it exactly that way — `file_content_sha256`
(single) stays Python; `file_content_sha256_batch` (the sweep) goes native.

## `m3-ingest` — directory walk (`fs_walk`)

`fs_walk` is a `read_dir` + `stat` sweep returning `(path, size, mtime, is_dir)`
with a cheap directory-ignore set and a symlink policy. It is the mechanical,
syscall-bound half of the Python walker; the nuanced filters (gitignore
semantics, binary sniff, filetype, glob) stay in Python. The win here is
**syscall-bound, not compute-bound**, so it is dominated by the OS and disk, not
by language — its value is parity + removing per-entry Python overhead on very
large trees rather than a fixed multiplier. Output parity vs `os.walk` (same file
set, `node_modules` pruned) is verified; raw wall-clock is filesystem-dependent
and intentionally not quoted as a headline number.

## `m3-governor` — pacing decision (`Governor.decide`)

The governor is a **pure decision function** (`load`, `elapsed` → pacing dict).
It is deliberately **not** a hot path — it runs once per pacing decision, not per
row — so this crate is **not** a performance optimization and no speedup is
claimed. Its value is single-source-of-truth (one pacing ladder, callable from
Python today and from future Rust daemons) and exact parity with the Python
`get_governor_pacing`. Parity is verified across the full truth table (10 ladder
cases + clamp/sanity edges) in `crates/m3-governor/tests/parity.rs` and again
native-vs-Python in m3-memory's `tests/test_governor_pacing.py`.

---

## Bonus finding — why a per-write commit queue is the wrong tool

While prototyping a write-batching layer for m3-memory, a multi-process
contention benchmark produced a result worth recording here because it informs
how this workspace should (and should not) be used:

| Scenario (separate processes, same SQLite DB) | per-write commit | 1 writer, batched txns | Speedup |
|---|---:|---:|---:|
| 8 procs × 250 writes (busy_timeout 50 ms) | 638 ms, 15 lock-retries | 13 ms, 0 retries | **48×** |
| 16 procs × 200 writes (busy_timeout 20 ms) | 1073 ms, 93 lock-retries | 20 ms, 0 retries | **54×** |
| 16 procs × 200 writes (busy_timeout 30 s) | 1302 ms, **0 retries** | 24 ms, 0 retries | 54× |

The takeaway: under genuine multi-writer contention, **batching commits** (not an
in-process queue) is what wins, and a sane `busy_timeout` already eliminates the
`database is locked` *errors* (it converts them into polite waits). m3-memory
acted on this by **reverting** an in-process write-queue prototype in favor of
its existing bulk-write APIs + `busy_timeout`. Full write-up:
m3-memory `docs/M3V3_OXIDATION.md` and `v3/m3_v3_phase_c_rust_oxidation_plan.md`.

---

## Reproducing

The vector/FTS/redaction suites live in m3-memory (`tests/bench_oxidation.py`,
`tests/bench_oxidation_fts_packed.py`) because they need the real
`agent_memory.db` vectors. The Milestone-4 micro-benchmarks above are simple
standalone scripts that build a synthetic tree / hammer a temp SQLite DB; rebuild
the wheel from current source (`crates/m3-core-py/build_local.py <backend>`) and
re-run before trusting any number, since a stale wheel silently runs the Python
fallback.
