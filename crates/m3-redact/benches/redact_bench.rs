//! Wave-0 baseline microbenches for `m3-redact`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use m3_redact::{RedactionConfig, Redactor};

fn full_config() -> RedactionConfig {
    RedactionConfig {
        enabled: true,
        patterns: vec![
            "api_keys".to_string(),
            "bearer_tokens".to_string(),
            "jwt".to_string(),
            "aws_keys".to_string(),
            "github_tokens".to_string(),
            "custom_regex".to_string(),
            "pii".to_string(),
        ],
        custom_regex: vec![
            r"\bproject-[a-z0-9]{8}\b".to_string(),
            r"\bcorp-id:\s*\d{6}\b".to_string(),
            r"(?i)\binternal-only\b".to_string(),
            r"\bbuild#\d{4,}\b".to_string(),
            r"\bticket-[A-Z]{2,4}-\d+\b".to_string(),
        ],
        redact_pii: true,
    }
}

fn make_4kb(seed_text: &str) -> String {
    let mut s = String::with_capacity(4096);
    while s.len() < 4096 {
        s.push_str(seed_text);
        s.push('\n');
    }
    s.truncate(4096);
    s
}

fn dirty_4kb_snippet() -> String {
    // 3 secrets + a few PII patterns embedded in lorem-ish text.
    let body = concat!(
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "Contact me at jane.doe@example.com or call (415) 555-0199. ",
        "AWS access key id AKIAIOSFODNN7EXAMPLE is in the env. ",
        "Auth: sk-ant-abcdefghij1234567890zzzzzzzz used for testing. ",
        "GitHub token ghp_123456789012345678901234567890abcdef. ",
        "More filler text describing nothing important at all, padding ",
        "the snippet so we hit a realistic 4KB chatlog turn size. ",
    );
    make_4kb(body)
}

fn clean_4kb_snippet() -> String {
    let body = concat!(
        "All systems nominal. Task completed without anomalies. ",
        "Throughput remains within the expected envelope and the queue ",
        "depth is shrinking as the worker pool drains. No alarms, no ",
        "warnings, no actions required from operators at this time. ",
    );
    make_4kb(body)
}

fn bench_compile(c: &mut Criterion) {
    let cfg = full_config();
    c.bench_function("redactor_new_full_config", |b| {
        b.iter(|| Redactor::new(black_box(&cfg)));
    });
}

fn bench_apply_dirty(c: &mut Criterion) {
    let r = Redactor::new(&full_config());
    let snippet = dirty_4kb_snippet();
    c.bench_function("redactor_apply_4kb_dirty", |b| {
        b.iter(|| r.apply(black_box(&snippet)));
    });
}

fn bench_apply_clean(c: &mut Criterion) {
    let r = Redactor::new(&full_config());
    let snippet = clean_4kb_snippet();
    c.bench_function("redactor_apply_4kb_clean", |b| {
        b.iter(|| r.apply(black_box(&snippet)));
    });
}

criterion_group!(benches, bench_compile, bench_apply_dirty, bench_apply_clean);
criterion_main!(benches);
