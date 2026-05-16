//! Wave-0 baseline microbenches for `m3-dispatcher`.
//!
//! GAP NOTE — `LengthBucketQueue::push` and `drain_one` are crate-private
//! (only `LengthBucketQueue::new` is `pub`). Benching the push/drain cycle
//! externally is therefore not possible without modifying production code,
//! which the wave-0 baseline harness explicitly forbids. The push/drain
//! microbench lives unimplemented; this gap is recorded in
//! `reports/bench_baseline.md`. We bench `estimate_tokens` here as the
//! sanity micro the spec calls for.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use m3_dispatcher::estimate_tokens;

fn bench_estimate_tokens(c: &mut Criterion) {
    // Deterministic 500-char string (no RNG needed; same string each iter).
    let s: String = "the quick brown fox jumps over the lazy dog. "
        .repeat(12); // ~540 chars
    let s = s[..500].to_string();
    c.bench_function("estimate_tokens_500ch", |b| {
        b.iter(|| estimate_tokens(black_box(&s)));
    });
}

criterion_group!(benches, bench_estimate_tokens);
criterion_main!(benches);
