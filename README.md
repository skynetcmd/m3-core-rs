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

`m3-core-py` builds as a Python wheel via `maturin`.
