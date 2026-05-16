//! Wave-0 baseline microbench for `m3-rank::fuse`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use m3_rank::{fuse, FusionWeights, RankRow, RankSource};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn build_inputs() -> (Vec<RankRow>, Vec<RankRow>) {
    let mut rng = StdRng::seed_from_u64(42);
    // 100 fts ids "fts_000".."fts_099", 100 vector ids drawn so ~30% overlap fts.
    let fts: Vec<RankRow> = (0..100)
        .map(|i| RankRow {
            id: format!("id_{i:03}"),
            score: rng.gen_range(0.0f32..30.0f32),
            source: RankSource::Fts,
        })
        .collect();
    // 30% overlap means 30 ids in [0..100), 70 ids in [100..170).
    let mut vector: Vec<RankRow> = Vec::with_capacity(100);
    for i in 0..30 {
        let idx = rng.gen_range(0..100);
        vector.push(RankRow {
            id: format!("id_{idx:03}"),
            score: rng.gen_range(0.0f32..1.0f32),
            source: RankSource::Vector,
        });
        let _ = i;
    }
    for i in 100..170 {
        vector.push(RankRow {
            id: format!("id_{i:03}"),
            score: rng.gen_range(0.0f32..1.0f32),
            source: RankSource::Vector,
        });
    }
    (fts, vector)
}

fn bench_fuse(c: &mut Criterion) {
    let (fts, vector) = build_inputs();
    let weights = FusionWeights { fts: 0.4, vector: 0.6 };
    c.bench_function("fuse_n100_n100_30pct_overlap", |b| {
        b.iter(|| {
            // Clone inside the closure: fuse takes Vec by value.
            fuse(black_box(fts.clone()), black_box(vector.clone()), black_box(weights))
        });
    });
}

criterion_group!(benches, bench_fuse);
criterion_main!(benches);
