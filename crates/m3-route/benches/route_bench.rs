//! Wave-0 baseline microbenches for `m3-route`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use m3_route::{decide_route, extract_signals};

/// Representative query mix — varied length, recency, entity-shape, intent.
const QUERIES: &[&str] = &[
    "logs",
    "what changed last week in the dispatcher",
    "Project Oxidation status update",
    "compare hybrid rank fusion against pure FTS5 retrieval on lme_m",
    "why did my embedding worker hang yesterday morning when I ran the bench",
];

fn bench_extract_signals(c: &mut Criterion) {
    c.bench_function("extract_signals_mix5", |b| {
        b.iter(|| {
            for q in QUERIES {
                black_box(extract_signals(black_box(q)));
            }
        });
    });
}

fn bench_decide_route(c: &mut Criterion) {
    let signals: Vec<_> = QUERIES.iter().map(|q| (*q, extract_signals(q))).collect();
    c.bench_function("decide_route_mix5", |b| {
        b.iter(|| {
            for (q, s) in &signals {
                black_box(decide_route(black_box(q), black_box(s)));
            }
        });
    });
}

criterion_group!(benches, bench_extract_signals, bench_decide_route);
criterion_main!(benches);
