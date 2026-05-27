# M3 Embedder Architecture

> How m3 routes embedding requests across the three available backends.
> Companion to `crates/m3-embed-server/README.md` (build + install
> mechanics) and `crates/m3-embed-llamacpp/` (the in-process llama.cpp
> wrapper). Read this if you're debugging "why is my embed slow / wrong /
> hung," configuring a new deployment, or evaluating m3 against
> alternatives.

---

## TL;DR

Three independent backends, each with a clear role:

| Backend | Mode | Concurrency | Use case |
|---|---|---:|---|
| **In-process Rust embedder** (tier 1) | per-process llama.cpp via `m3-embed-llamacpp` | 8–20+ streams | Hot path in long-lived processes — Python MCP server, agentic loops, bench harnesses |
| **m3-embed-server** (tier 2) | always-on HTTP service on `:8082` | 2 streams | Cold-start fallback; ensures any new MCP server process has a working embedder without per-process GGUF load |
| **Primary HTTP failover** (tier 3) | OpenAI-shape endpoint (LM Studio, Ollama, llama-server, etc.) | depends on backend | Last-resort. May serve non-BGE-M3 vectors → cross-space cosines if cascade misroutes here. |

Default cascade: try tier 1 → tier 2 → tier 3. m3-memory's `_embed`
honors this; clients that hit `m3-embed-server` directly skip tier 1
and start at tier 2.

---

## Why three backends

Each solves a different deployment constraint:

**Tier 1 (in-proc)** is the fastest path. A long-lived process loads the
GGUF once into VRAM/RAM and reuses the model across thousands of embed
calls. Zero IPC, zero HTTP serialization. Throughput on a modern GPU is
1k–10k embeds/sec at `streams=20`. But it costs ~6–16 GB of memory
per-process, so spawning short-lived workers all loading the same GGUF
is wasteful.

**Tier 2 (HTTP `:8082`)** solves the "short-lived process / no GGUF
configured" problem. One copy of the model lives in the
`m3-embed-server` service; any number of clients hit it over HTTP. CPU-
only by design (so it stays small in `streams=2` and never fights the
GPU for VRAM). Throughput is ~100 embeds/sec for typical BGE-M3 inputs
on a recent x86 CPU. Always-on via Windows Service / launchd / systemd.

**Tier 3 (primary HTTP failover)** is the legacy / interop path. m3
will probe LM Studio (`:1234`), Ollama (`:11434`), and other
OpenAI-shape endpoints. **Warning**: those servers may NOT serve BGE-M3
— a Ollama setup might return a 4096-dim Llama embedding, which is
cross-space-incompatible with BGE-M3 vectors already in the m3 index.
The cascade in m3-memory now prefers tier 2 over tier 3 (commit
`0dfdf56`) precisely because tier 3 is hard to validate. Tier 3 should
only fire when both tier 1 and tier 2 are unavailable — and the
operator should treat that as a deployment alarm, not normal operation.

---

## Cascade ordering and failure modes

The Python m3-memory cascade (`bin/memory/embed.py`) goes:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Tier 1 — in-process Rust embedder                            │
│    Gate: m3_core_rs imported AND M3_EMBED_GGUF set AND          │
│           breaker closed                                         │
│    Failure → log warning, fall through                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Tier 2 — m3-embed-server HTTP at M3_EMBED_FALLBACK_URL       │
│    Gate: breaker closed                                          │
│    Failure → log warning, fall through                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Tier 3 — primary HTTP via llm_failover                       │
│    Gate: breaker closed; probes LM Studio :1234, Ollama :11434  │
│    Failure → return (None, EMBED_MODEL)                         │
└─────────────────────────────────────────────────────────────────┘
```

Each tier has a circuit breaker (3-failure threshold, configurable
reset timeout). Once tripped, the cascade skips that tier for the
reset window. This bounds total wall-clock on a multi-tier outage to
roughly `threshold * timeout`.

### Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `memory_search` hangs >10s | Tier 1 unset AND tier 2 not installed → tier 3 retries with backoff | `m3-embed-server install && start`, or set `M3_EMBED_GGUF` |
| Returned vectors look wrong / cross-space | Tier 3 served Ollama / Llama embeddings instead of BGE-M3 | Same as above; m3 cascade now prefers tier 2 over tier 3 |
| Tier 1 OOM / CUDA error | `streams` too high for GPU memory | Lower `M3_EMBED_STREAMS` (default 8 for 16GB GPU; 20+ for 24GB+) |
| First call after install fails | SCM/launchd hasn't started service yet | `m3-embed-server install` now polls `/health` for 10s post-install (B3) |
| Doctor says GGUF missing | `M3_EMBED_GGUF` unset and discovery cascade found nothing | Either set the env var or drop a BGE-M3 GGUF into one of the discovery dirs (see `m3-embed-server doctor`) |

---

## GGUF discovery cascade (B5)

`m3-embed-server` resolves the GGUF path in priority order:

1. `M3_EMBED_GGUF` env var
2. `[embed].gguf` in the config.toml (`%PROGRAMDATA%\m3-embed-server\`
   on Windows; `~/.config/m3-embed-server/` on Linux;
   `~/Library/Application Support/m3-embed-server/` on macOS)
3. **Discovery cascade**: walk these directories (up to depth 4) looking
   for any file matching `*bge[-_]m3*.gguf` (case-insensitive):
   - `~/.lmstudio/models/` (most developer boxes have this)
   - `~/Library/Application Support/LM Studio/models/` (macOS legacy)
   - `~/.cache/m3/models/` (populated by `fetch_sovereign_assets.py`)
   - `~/.m3-memory/_assets/embedder/`
   - `~/models/`
4. Fail — error message names every dir tried

This is intentional: a developer who already has BGE-M3 from another
agent shouldn't have to specify the path. `m3-embed-server doctor`
prints the cascade and which step won.

---

## Service lifecycle (B4)

`m3-embed-server` writes a `config.toml` on first foreground start if
none exists, capturing the resolved config (post-discovery). This:

- Makes the install → run → reproduce path transparent (config file
  reflects what the binary actually used, not just what the env said).
- Survives env-var changes — a service install snapshot stays stable
  even after the operator removes the env var that originally set GGUF.
- Gives post-mortems something to read.

Manual config edits to the TOML file take effect on next start.

---

## Doctor subcommand (B1)

```bash
m3-embed-server doctor
```

Runs six probes (each bounded; total wall-clock < 15s even with
service down):

1. config.toml presence + parse
2. GGUF resolution path + discovery cascade dirs (which exist)
3. OS service status (Windows SCM / launchd / systemd)
4. HTTP `/health` on configured host/port
5. HTTP `/v1/embeddings` roundtrip with verification of returned dim
6. Recent log-file lines (tail of last 20)

Exit code 0 on overall pass; 1 if any critical probe failed. The output
is grep-friendly so CI / install scripts can parse it. Use this before
opening a bug or migrating deployments.

---

## Cross-references

- `crates/m3-embed-server/README.md` — build, install, OS-service
  mechanics
- `crates/m3-embed-llamacpp/src/lib.rs` — in-process backend
  implementation
- `crates/m3-dispatcher/` — the streams=N batching scheduler that
  powers tier-1 concurrency
- m3-memory `bin/memory/embed.py` — Python cascade implementation,
  tier ordering and breakers
- m3-memory `bin/memory/doctor.py` — Python-side companion to the
  Rust `doctor` subcommand; probes tier 1 GGUF state + tier 2 HTTP
  from inside the MCP server
