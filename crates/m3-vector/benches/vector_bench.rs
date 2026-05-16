//! Wave-0 baseline microbenches for `m3-vector`.
//!
//! Deterministic: all RNG seeded from `StdRng::seed_from_u64(42)`. Inputs are
//! wrapped in `criterion::black_box` to defeat optimizer hoisting.
//!
//! Notes:
//! - `cosine_unchecked` is private. The "single-call cosine" bench therefore
//!   exercises the public `cosine()` (which performs a length check + dispatch
//!   to the same inner loop). That adds a constant-time dim check, not loop
//!   work, so the measured median is still dominated by the 1024-dim accumulate.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use m3_vector::{
    cosine, cosine_batch_packed, f32_as_blob, hybrid_score_batch, mmr_rerank_scored,
    recency_bonus_ranks,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEED: u64 = 42;
const DIM: usize = 1024;

fn rand_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect()
}

fn bench_cosine_single(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let a = rand_vec(&mut rng, DIM);
    let b = rand_vec(&mut rng, DIM);
    c.bench_function("cosine_1024", |bencher| {
        bencher.iter(|| {
            // black_box defeats hoisting of the inputs out of the loop.
            cosine(black_box(&a), black_box(&b)).unwrap()
        });
    });
}

fn bench_cosine_batch_packed(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let query = rand_vec(&mut rng, DIM);
    let vecs: Vec<Vec<f32>> = (0..1000).map(|_| rand_vec(&mut rng, DIM)).collect();
    // Pack each vector into its on-disk f32-LE blob form.
    let blobs: Vec<Vec<u8>> = vecs.iter().map(|v| f32_as_blob(v).to_vec()).collect();
    c.bench_function("cosine_batch_packed_1000x1024", |bencher| {
        bencher.iter(|| {
            let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
            cosine_batch_packed(black_box(&query), black_box(&refs), DIM).unwrap()
        });
    });
}

fn bench_hybrid_score_batch(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 200;
    let vector_scores: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
    let bm25_scores: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..20.0f32)).collect();
    let content_lens: Vec<u32> = (0..n).map(|_| rng.gen_range(1u32..2000u32)).collect();
    let importances: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
    let title_overlaps: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
    c.bench_function("hybrid_score_batch_n200", |bencher| {
        bencher.iter(|| {
            hybrid_score_batch(
                black_box(&vector_scores),
                black_box(&bm25_scores),
                black_box(&content_lens),
                black_box(&importances),
                black_box(&title_overlaps),
                0.7,
                0.1,
                0.15,
                40,
            )
            .unwrap()
        });
    });
}

fn bench_mmr_rerank_scored(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 200;
    let relevance: Vec<f32> = {
        // Pre-sort descending so force_seed_first picks the genuine top item.
        let mut r: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0f32..1.0f32)).collect();
        r.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        r
    };
    let vecs: Vec<Vec<f32>> = (0..n).map(|_| rand_vec(&mut rng, DIM)).collect();
    c.bench_function("mmr_rerank_scored_k20_n200_d1024", |bencher| {
        bencher.iter(|| {
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            mmr_rerank_scored(black_box(&relevance), black_box(&refs), 0.7, 20, true).unwrap()
        });
    });
}

fn bench_recency_bonus_ranks(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let n = 500;
    // 80% Some(ISO date) in 2025-01-01..2026-12-31 range, 20% None.
    let dates: Vec<Option<String>> = (0..n)
        .map(|_| {
            if rng.gen_bool(0.8) {
                let year = if rng.gen_bool(0.5) { 2025 } else { 2026 };
                let month = rng.gen_range(1u32..=12);
                let day = rng.gen_range(1u32..=28);
                Some(format!("{year:04}-{month:02}-{day:02}"))
            } else {
                None
            }
        })
        .collect();
    c.bench_function("recency_bonus_ranks_n500", |bencher| {
        bencher.iter(|| recency_bonus_ranks(black_box(&dates), 1.0));
    });
}

criterion_group!(
    benches,
    bench_cosine_single,
    bench_cosine_batch_packed,
    bench_hybrid_score_batch,
    bench_mmr_rerank_scored,
    bench_recency_bonus_ranks,
);
criterion_main!(benches);
