# macOS Metal build — m3-core-rs 3.6.22 (run on the Apple-Silicon Mac)

Goal: build the 4 missing `m3-core-rs-macos-metal` wheels (cp311–314) to complete
the `v2026.06.22` GitHub release (currently 24/28 — Windows 12 + Linux 12).

macOS is **Metal-only** (`embedded-metal`), so it already ships an in-process
BGE-M3 `EmbeddedEmbedder` — the CPU-embedder policy change doesn't alter macOS.
Wheels are ~2.6 MB (Metal.framework is system-provided, not bundled).

---

## 0. One-time toolchain setup (skip if already provisioned)

```bash
# Homebrew Pythons (3.11 is NOT pre-installed — add it explicitly), cmake, Rust
brew install python@3.11 python@3.12 python@3.13 python@3.14 cmake rustup-init
rustup-init -y && source "$HOME/.cargo/env"
pipx install maturin            # maturin >=1.7,<2
# sanity
rustc --version && maturin --version && cmake --version
```

## 1. Sync the repo to the 3.6.22 commit

```bash
cd ~/m3-memory/m3-core-rs            # the macOS repo clone path (per ops playbook)
git checkout main && git pull origin main
grep -m1 '^version' Cargo.toml      # MUST print: version = "3.6.22"
```

> If that does NOT say 3.6.22, stop — do not build a mislabeled wheel.

## 2. Build all 4 Metal wheels (cache-optimal driver)

```bash
# build_local.py resolves cp311–314 via uv/brew, builds metal, smoke-tests on the GPU.
python crates/m3-core-py/build_local.py metal
```

Equivalent explicit form (only if build_local.py can't resolve interpreters):

```bash
python crates/m3-core-py/build_wheel.py --backend metal --os macos --release \
    --out dist/m3-core-rs-macos-metal \
    -- --interpreter python3.11 python3.12 python3.13 python3.14
```

> ALWAYS use build_local.py / build_wheel.py — never bare `maturin build`. The
> script rewrites `[project].name` to `m3-core-rs-macos-metal`; a bare build is
> named wrong and the install wizard won't find it.

## 3. Verify the wheels (per backend correctness contract)

Expect 4 files like `m3_core_rs_macos_metal-3.6.22-cp31X-cp31X-macosx_11_0_arm64.whl`,
~2.6 MB each. Spot-check one in a throwaway venv (needs a bge-m3 GGUF on the Mac):

```bash
WHL=$(ls ci-wheels/local-3.6.22/macos-metal/*cp313*.whl 2>/dev/null || ls dist/m3-core-rs-macos-metal/*cp313*.whl)
python3.13 -m venv /tmp/m3mac && /tmp/m3mac/bin/python -m pip install --no-deps "$WHL"
/tmp/m3mac/bin/python - <<'PY'
import m3_core_rs as m, math
print("EmbeddedEmbedder:", hasattr(m, "EmbeddedEmbedder"))     # expect True
print("backend_label:", m.embed_backend_label())               # expect 'metal'
print("8/8 funcs:", all(hasattr(m,f) for f in
  ["sanitize_fts","compile_fts_query","token_jaccard","token_jaccard_batch",
   "cosine_batch_packed","mmr_rerank_scored_packed","rank_hybrid_packed","scrub"]))
# end-to-end (point at a bge-m3 GGUF on this Mac):
GGUF="$HOME/.lmstudio/models/.../bge-m3-GGUF-Q4_K_M.gguf"   # <-- set the real path
e = m.EmbeddedEmbedder(GGUF); v = e.embed(["macos metal embed test"])[0]
print("dim", len(v), "L2 %.4f" % math.sqrt(sum(x*x for x in v)))  # expect 1024, ~1.0
PY
rm -rf /tmp/m3mac
```

Healthy result: `EmbeddedEmbedder True`, `backend_label metal`, `8/8 funcs True`,
`dim 1024`, `L2 ~1.0`.

## 4. Get the 4 wheels onto the upload box and attach to the release

Use whichever box holds the `gh` auth for `skynetcmd/m3-core-rs`. From the Mac,
copy the wheels there (adjust host/path), or run `gh` on the Mac if it's authed:

```bash
# from the Mac → the upload box (example; use your own scp alias/path):
scp ci-wheels/local-3.6.22/macos-metal/*.whl <upload-host>:<repo>/ci-wheels/macos-3.6.22/
```

Then on the box with `gh` auth:

```bash
gh release upload v2026.06.22 --repo skynetcmd/m3-core-rs \
    <path>/m3_core_rs_macos_metal-3.6.22-*.whl --clobber
gh release view v2026.06.22 --repo skynetcmd/m3-core-rs \
    --json assets --jq '.assets|length'        # MUST be 28
```

## 5. Publish (only when 28/28)

```bash
# Gate: do NOT publish until the asset count is 28.
gh release edit v2026.06.22 --repo skynetcmd/m3-core-rs --draft=false
```

Version pins (`Cargo.toml` workspace + `m3-memory/m3_memory/rust_core_install.py`)
are already bumped to 3.6.22 / v2026.06.22 — no further lockstep edit needed.

---

### Notes / gotchas
- macOS leg was authored in CI but historically **not verified on a device** — this
  is the verification run; if anything fails, capture the build log
  (`ci-wheels/local-3.6.22/build-macos-metal.log`) before retrying.
- Metal builds **native** (no zig; zig is Linux-CPU-only).
- Wheels are `.gitignore`d (`*.whl`) — never `git add` them.
