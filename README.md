# m3-core-rs

Rust compute core for m3-memory ("Project Oxidation"). A Cargo workspace of focused,
independently-useful crates plus a PyO3 binding crate (`m3-core-py`) consumed by m3-memory.

## Crates

| Crate | Phase | Purpose |
|---|---|---|
| `m3-error` | 1.2 | Shared `M3Error` type with Python exception mapping |
| `m3-hash` | 1.3 / 3d | FIPS-preserving SHA-256 hashing (`ring`-backed) |
| `m3-vector` | 2 | SIMD cosine similarity, MMR reranker |
| `m3-dispatcher` | 3b/3c | Generic `ModelBackend` coalescer (length bucketing, backpressure) |
| `m3-embed-llamacpp` | 3b | llama.cpp embedding backend |
| `m3-ner-ort` | 3c | ONNX Runtime GLiNER NER backend |
| `m3-redact` | 3d | Multi-pattern secret redaction |
| `m3-rank` | 3d | Hybrid FTS5 + vector rank-fusion merge |
| `m3-route` | 3d | Multi-signal query route decider |
| `m3-graph` | 4 | In-memory graph index + traversal |
| `m3-fts` | 4 | FTS5 query sanitizer + lexical tokenizer |
| `m3-governor` | 4 | Adaptive background-workload pacing logic (pure decision function) |
| `m3-ingest` | 4 | Filesystem-walker hot path: parallel directory walk + batch content hashing |
| `m3-embed-server` | 3b | OpenAI-compatible HTTP embedding server (the shared-embedder baseline; one server, many thin clients on :8082) |
| `m3-core-py` | — | PyO3 bindings; the only crate Python sees |

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for per-operation speed measurements.

## Reusability

Generic crates use generic names in their public API. The `M3` prefix and the `M3_*`
env-var convention live only in `m3-core-py`. A pure-Rust user can depend on any generic
crate without knowing m3-memory exists.

## Build

```
cargo build --workspace
cargo test --workspace
```

`m3-core-py` builds as a Python wheel via `maturin`. See
[`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the per-machine release workflow
and [`docs/BUILD_WHEELS.md`](docs/BUILD_WHEELS.md) for building your own.

### Wheel matrix

m3-core-rs publishes **7 packages** (one per OS × backend) across **CPython
3.11–3.14** → a full release is **28 wheels**. Every wheel bundles **both**
native artifacts: the in-process `EmbeddedEmbedder` (`m3_core_rs.*.{pyd,so}`)
**and** the `m3-embed-server` shared-server binary
(`m3_core_rs/m3-embed-server[.exe]`) — see `crates/m3-core-py/build_wheel.py`.
`crates/m3-core-py/verify_wheels.py` asserts both are present, backend-matched,
and RECORD-correct.

The bundled `m3-embed-server` size tracks the backend, because GPU backends
statically link a GPU-accelerated llama.cpp while **Metal links Apple's system
framework dynamically** (so a Metal server is CPU-sized):

| OS | cpu | vulkan | cuda | metal |
|---|---|---|---|---|
| **Windows** (`win_amd64`) | 7.6 MiB | 65.7 MiB | 138.4 MiB | — |
| **Linux** (`manylinux`) | 8.3 MiB | 49.4 MiB | 621.5 MiB | — |
| **macOS** (`macosx_11_0_arm64`, Apple Silicon) | — | — | — | 7.8 MiB |

Sizes are the embedded `m3-embed-server` binary (MiB = 1024², as reported by
`verify_wheels.py`), measured for the 3.7.4 release — all 28 wheels verified.
macOS is Metal-only by design (Apple Silicon always has a Metal GPU).

> **Linux-CUDA is large (~622 MiB binary → ~660 MB wheel).** The Linux CUDA build
> statically embeds SASS+PTX kernels for every supported compute capability
> (sm_75 → sm_121a); the `.text` section alone is ~122 MB (the binary is already
> stripped — this is real kernel code, not debug symbols). It is ~4.5× the
> Windows-CUDA binary and exceeds PyPI's default 100 MB per-file limit — publishing
> requires a limit increase, or a size reduction (trim the compute-capability list,
> or split PTX-only vs SASS builds).
