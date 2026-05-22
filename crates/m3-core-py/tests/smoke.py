"""Smoke test for the m3_core_rs PyO3 bindings.

Exercises every binding with assert-based checks. NOT run by `cargo test` —
pyo3 extension modules need the Python interpreter.

How to run:
    cd crates/m3-core-py
    maturin develop          # builds + installs m3_core_rs into the active venv
    python tests/smoke.py

Status: written but unverified-by-execution in this environment (maturin not
available here). Build correctness is covered by `cargo build -p m3-core-py`.
"""

import os
import struct

import m3_core_rs as m


def test_hash() -> None:
    h = m.sha256_hex("hello")
    assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", h
    assert isinstance(m.hash_provider(), str)


def test_vector() -> None:
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert abs(m.cosine(a, b) - 1.0) < 1e-6
    assert abs(m.cosine(a, c)) < 1e-6
    batch = m.cosine_batch(a, [b, c])
    assert len(batch) == 2 and abs(batch[0] - 1.0) < 1e-6
    sel = m.mmr_rerank(a, [b, c, a], 0.5, 2)
    assert len(sel) == 2 and all(isinstance(i, int) for i in sel)
    blob = m.f32_as_blob([1.5, 2.5])
    assert blob == struct.pack("<2f", 1.5, 2.5)
    assert m.blob_as_f32(blob) == [1.5, 2.5]


def test_redact() -> None:
    cfg = {
        "enabled": True,
        "patterns": ["pii"],
        "custom_regex": [],
        "redact_pii": True,
    }
    scrubbed, n, groups = m.scrub("contact me at user@example.com please", cfg)
    assert "[REDACTED:pii]" in scrubbed
    assert n == 1
    assert groups == ["pii"]
    scrubbed2, n2, groups2 = m.scrub("nothing here", cfg)
    assert n2 == 0 and groups2 == []
    # disabled config is a no-op
    off = {"enabled": False, "patterns": [], "custom_regex": [], "redact_pii": False}
    assert m.scrub("user@example.com", off) == ("user@example.com", 0, [])
    # bad custom regex doesn't crash; error is retrievable
    bad = {
        "enabled": True,
        "patterns": ["custom_regex"],
        "custom_regex": ["[unclosed"],
        "redact_pii": False,
    }
    m.scrub("irrelevant", bad)
    assert m.redaction_compile_errors()


def test_rank() -> None:
    fts = [m.RankRow("a", 10.0, "fts"), m.RankRow("b", 5.0, "fts")]
    vec = [m.RankRow("a", 1.0, "vector"), m.RankRow("c", 9.0, "vector")]
    fused = m.fuse(fts, vec, 0.5, 0.5)
    ids = {row.id for row in fused}
    assert ids == {"a", "b", "c"}
    assert all(0.0 <= row.score <= 1.0 for row in fused)
    # descending order
    assert fused == sorted(fused, key=lambda r: -r.score)


def test_route() -> None:
    sig = m.extract_signals("What did I do yesterday?")
    assert sig.recency_cue and sig.question_form
    assert sig.token_count == 5
    dec = m.decide_route("What did I do yesterday?", sig)
    assert dec.branch == "temporal"
    assert 0.0 <= dec.confidence <= 1.0
    assert len(dec.signal_breakdown) == 4


def test_graph() -> None:
    g = m.GraphIndex()
    g.add_node("x")
    g.add_edge("x", "y", "rel", 1.0)
    g.add_edge("y", "z", "rel", 1.0)
    assert g.node_count() == 3
    assert g.edge_count() == 2
    nb = g.neighbors_within("x", 2)
    assert ("x", 0) in nb and ("z", 2) in nb
    path = g.shortest_path("x", "z")
    assert path == ["x", "y", "z"]
    assert g.shortest_path("z", "x") is None
    exp = g.expand(["x"], 2, 10)
    assert set(exp) == {"y", "z"}


def test_resilience() -> None:
    cb = m.CircuitBreaker(2, 0.05)
    assert cb.state() == "closed"
    assert cb.allow_request()
    cb.record_failure()
    cb.record_failure()
    assert cb.state() == "open"
    cb.record_success()
    assert cb.state() == "closed"

    rp = m.RetryPolicy(5, 0.1, 2.0)
    assert rp.max_attempts == 5
    assert abs(rp.delay_for_attempt(1) - 0.1) < 1e-6
    assert abs(rp.delay_for_attempt(3) - 0.4) < 1e-6
    assert rp.delay_for_attempt(20) <= 2.0


def test_ner() -> None:
    assert m.ep_priority("windows") == ["DirectML", "CPU"]
    assert m.ep_priority("cpu_only") == ["CPU"]
    # shape (2 spans, 1 width, 2 labels)
    spans = m.decode_spans([0.9, 0.1, 0.2, 0.8], (2, 1, 2), 0.5)
    assert len(spans) == 2
    assert spans[0].start == 0 and spans[0].label == 0
    assert spans[1].start == 1 and spans[1].label == 1


def test_dispatcher_and_env() -> None:
    assert m.estimate_tokens("a" * 40) == 10
    assert m.estimate_tokens("") == 1

    # defaults (assuming no M3_* env vars set)
    cfg = m.DispatcherConfig()
    assert cfg.streams >= 1
    assert len(cfg.length_buckets) > 0

    # kwarg override beats env/default (precedence rule §9.7)
    cfg2 = m.DispatcherConfig(streams=32, coalesce_window_ms=7)
    assert cfg2.streams == 32
    assert cfg2.coalesce_window_ms == 7

    summary = m.env_config_summary()
    for key in (
        "M3_EMBED_STREAMS",
        "M3_EMBED_COALESCE_MS",
        "M3_EMBED_MAX_BATCH_TOKENS",
        "M3_EMBED_CTX",
        "M3_EMBED_N_BATCH",
        "M3_EMBED_N_UBATCH",
        "M3_HASH_PROVIDER",
    ):
        assert key in summary, key

    # Regression guard: n_ubatch / n_batch must default to n_ctx. The BERT
    # encoder asserts `n_ubatch >= n_tokens` over the whole decode batch, and a
    # chunk can be as large as n_ctx — so a default below n_ctx (the old 512 /
    # 2048) crashes the encoder on routine BGE-M3 inputs. See to-do a57dac19.
    assert summary["M3_EMBED_N_UBATCH"] == summary["M3_EMBED_CTX"], (
        "M3_EMBED_N_UBATCH default must equal M3_EMBED_CTX (encoder-safe); "
        f"got n_ubatch={summary['M3_EMBED_N_UBATCH']} ctx={summary['M3_EMBED_CTX']}"
    )
    assert summary["M3_EMBED_N_BATCH"] == summary["M3_EMBED_CTX"], (
        "M3_EMBED_N_BATCH default must equal M3_EMBED_CTX; "
        f"got n_batch={summary['M3_EMBED_N_BATCH']} ctx={summary['M3_EMBED_CTX']}"
    )

    # Regression guard: the dispatcher's coalescing ceiling must default to
    # streams * ctx so one coalescing window can fill the whole worker pool.
    # The earlier flat 2048 under-fed the pool by a factor of `streams`.
    assert (
        summary["M3_EMBED_MAX_BATCH_TOKENS"]
        == summary["M3_EMBED_STREAMS"] * summary["M3_EMBED_CTX"]
    ), (
        "M3_EMBED_MAX_BATCH_TOKENS default must equal STREAMS * CTX; got "
        f"{summary['M3_EMBED_MAX_BATCH_TOKENS']} vs "
        f"{summary['M3_EMBED_STREAMS']} * {summary['M3_EMBED_CTX']}"
    )

    # n_ubatch / n_batch track n_ctx when ctx is raised via env; and
    # max_batch_tokens recomputes from the new ctx too.
    os.environ["M3_EMBED_CTX"] = "4096"
    bumped = m.env_config_summary()
    assert bumped["M3_EMBED_N_UBATCH"] == 4096, bumped["M3_EMBED_N_UBATCH"]
    assert bumped["M3_EMBED_N_BATCH"] == 4096, bumped["M3_EMBED_N_BATCH"]
    assert (
        bumped["M3_EMBED_MAX_BATCH_TOKENS"]
        == bumped["M3_EMBED_STREAMS"] * 4096
    ), bumped["M3_EMBED_MAX_BATCH_TOKENS"]
    del os.environ["M3_EMBED_CTX"]

    # An explicit M3_EMBED_MAX_BATCH_TOKENS still overrides the computed default.
    os.environ["M3_EMBED_MAX_BATCH_TOKENS"] = "1024"
    assert m.env_config_summary()["M3_EMBED_MAX_BATCH_TOKENS"] == 1024
    del os.environ["M3_EMBED_MAX_BATCH_TOKENS"]

    # env var is picked up when set
    os.environ["M3_EMBED_STREAMS"] = "16"
    assert m.env_config_summary()["M3_EMBED_STREAMS"] == 16
    assert m.DispatcherConfig().streams == 16
    # kwarg still wins over env
    assert m.DispatcherConfig(streams=4).streams == 4
    del os.environ["M3_EMBED_STREAMS"]


def main() -> None:
    tests = [
        test_hash,
        test_vector,
        test_redact,
        test_rank,
        test_route,
        test_graph,
        test_resilience,
        test_ner,
        test_dispatcher_and_env,
    ]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} smoke tests passed")


if __name__ == "__main__":
    main()
