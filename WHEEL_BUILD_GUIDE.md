# m3-core-rs — Wheel Build & Deployment Guide

How to build, verify, and publish the `m3-core-rs` Python wheels across
**Linux, Windows, and macOS**. This is the operational companion to
[`docs/PUBLISHING.md`](docs/PUBLISHING.md) (which covers the PyPI / trusted-
publisher side) and [`crates/m3-core-py/build_wheel.py`](crates/m3-core-py/build_wheel.py)
(the single source of truth for the package-name + feature mapping).

> This is the **public, generic** how-to. The maintainers keep a private ops
> playbook (concrete build-host names/access, per-release status, and prior
> decisions) outside this repo; the two cross-reference each other.

> **TL;DR**: `m3-core-rs` is one Rust source tree published as **7 differently
> named PyPI projects** — one per `(OS, backend)` — each containing **one wheel
> per supported CPython** (3.11–3.14). All install the same `m3_core_rs` import
> module. CI (`.github/workflows/release.yml`) builds the matrix on native
> runners and publishes via PyPI Trusted Publishing. Wheels are **never** committed
> to git.

---

## 1. The model — why 7 projects × 4 Pythons

`m3-core-rs` is **one crate** but ships as several PyPI packages because the GPU
backend is compiled in, and a CUDA wheel can't run on a Vulkan/CPU host. So the
`(OS, backend)` pair is baked into the **PyPI project name**, and the **Python
version** rides in the wheel filename's `cpXY` compatibility tag.

| OS | Backend | PyPI project | Cargo features | `EmbeddedEmbedder`? |
|----|---------|--------------|----------------|----------------------|
| Linux | CPU | `m3-core-rs-linux-cpu` | `embedded` | ✅ yes (CPU llama.cpp) |
| Linux | CUDA | `m3-core-rs-linux-cuda` | `embedded-cuda` | ✅ yes (NVIDIA) |
| Linux | Vulkan | `m3-core-rs-linux-vulkan` | `embedded-vulkan` | ✅ yes (any Vulkan GPU) |
| Windows | CPU | `m3-core-rs-windows-cpu` | `embedded` | ✅ yes (CPU llama.cpp) |
| Windows | CUDA | `m3-core-rs-windows-cuda` | `embedded-cuda` | ✅ yes |
| Windows | Vulkan | `m3-core-rs-windows-vulkan` | `embedded-vulkan` | ✅ yes |
| macOS (Apple Silicon) | Metal | `m3-core-rs-macos-metal` | `embedded-metal` | ✅ yes |

### Two things that trip people up

1. **Every backend — including CPU — ships an in-process `EmbeddedEmbedder`.**
   CPU uses the plain `embedded` feature (CPU-only llama.cpp, no GPU backend);
   the GPU backends use `embedded-cuda`/`-vulkan`/`-metal`. This is a deliberate
   policy: **m3-memory must always have a default in-process BGE-M3 embedder on
   every build** — an embedder-less CPU wheel forced reliance on the embed-server
   (tier 2), which is not guaranteed running on an offline host, leaving a gap
   where embedding could return nothing. `m3_memory/rust_core_install.py` mirrors
   this: `_BACKEND_FEATURES["cpu"] = ["embedded"]`. **A CPU wheel *without*
   `EmbeddedEmbedder` is now built wrong** (it was built with `default = []` /
   no `--features embedded`). Trade-off: CPU builds now cmake-compile llama.cpp,
   so they need a C/C++ compiler + cmake (no longer toolchain-free), and the CPU
   wheel grows from ~1 MB to ~2.4 MB — still far below the GPU wheels (20–122 MB).
   Verified 2026-06-22: a `--features embedded` Windows CPU wheel loads BGE-M3
   in-process (dim 1024, L2-norm 1.0, `embed_backend_label()` == `cpu`).

2. **Python versions are NOT separate projects, tags, or publishers.** One
   project + version holds 4 wheels (`...-cp311-...`, `-cp312-`, `-cp313-`,
   `-cp314-`). pip picks the match. You never make a per-Python tag or
   publisher — that fights the ecosystem. The git tag (`v2026.7.25`) stores
   the **package version** (`3.7.25`) only.

### Naming — use `build_wheel.py`, nothing else

`maturin` has no `--name` flag; the wheel name comes from `[project].name`.
`build_wheel.py` temporarily rewrites that to `m3-core-rs-<os>-<backend>` around
the maturin call. **Always build through it.** A wheel built with a bare
`maturin build` is named `m3_core_rs-...` (project `m3_core_rs`) and the m3
wizard's `pip install m3-core-rs-<os>-<backend>==<ver>` will **not** find it.

> ⚠️ **Do NOT** use a `+cpu` / `+vulkan` PEP 440 *local version* label
> (e.g. `m3_core_rs-3.7.25+vulkan-...`). PyPI **rejects local-version wheels on
> upload**, and it doesn't match the wizard's per-project install. Any such
> wheels currently attached to a GitHub Release are non-canonical and predate
> this guide — re-build them through `build_wheel.py`.

---

## 2. Supported Python versions

m3-memory declares `requires-python >= 3.11`; the wheel matrix covers
**3.11, 3.12, 3.13, 3.14**. Build all four for every package. A user on a Python
outside this range gets a source-build fallback (needs Rust + a compiler) or the
CPU embed-server path — functional but slow, so keep the prebuilt set complete.

Debian/most distros don't ship every version. The clean way to get them without
polluting the system Python is **[`uv`](https://docs.astral.sh/uv/)**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11 3.12 3.13 3.14    # standalone CPython builds in ~/.local
```

---

## 3. Building per platform

All commands run from the repo root unless noted. The canonical invocation is:

```bash
python crates/m3-core-py/build_wheel.py \
    --backend <cpu|cuda|vulkan|metal> \
    --os <linux|windows|macos> \
    --out dist/<project> \
    -- --interpreter python3.11 python3.12 python3.13 python3.14
```

Everything after `--` is forwarded verbatim to `maturin build`.

### Optimal build order & caching (build the whole matrix fast)

You ship `backends × {3.11, 3.12, 3.13, 3.14}` wheels. The expensive artifact in
every build is the **cmake/C++ compile of llama.cpp + ggml** (and the GPU shader
toolchains). The key fact that makes the matrix cheap:

> **llama.cpp is Python-version-independent.** It's a pure C/C++ library that
> does not link Python. Only the thin PyO3 binding (`m3-core-py`'s `cdylib`)
> links a specific CPython ABI. So across Python versions of the *same* backend,
> llama.cpp is compiled **once** and reused; only the small binding relinks.

Two rules fall out of this:

1. **One `build_wheel.py` call per backend, with all four interpreters at once.**
   Pass `--interpreter <py311> <py312> <py313> <py314>` in a single invocation.
   maturin builds the Rust/C artifacts once and fans out the per-interpreter
   binding — far better cache reuse than four separate calls.

2. **Loop backends on the OUTER level, Python versions on the inner.** cargo keys
   its `target/` cache on the *feature set*, not the interpreter. CPU=`embedded`,
   Vulkan=`embedded-vulkan`, CUDA=`embedded-cuda` are different features, so each
   backend switch forces exactly one llama.cpp rebuild — unavoidable, but you pay
   it **once per backend (3×)**, not once per (backend, version) (12×). Sweeping
   versions inside a fixed backend keeps llama.cpp cached. **Never** loop
   backends inside a Python-version loop — that is the 12× pessimal order.

```
for backend in cpu vulkan cuda:          # OUTER: each switch = 1 llama.cpp rebuild
    build_wheel.py --backend $backend ... -- --interpreter py311 py312 py313 py314
                                         # INNER: 4 fast binding relinks, llama.cpp cached
```

Verified 2026-06-22 (Windows, crate 3.6.6): built all three backends × cp311–314
(12 wheels) this way — within each backend only the first Python version paid the
llama.cpp compile; the remaining three were quick binding relinks.

**`build_local.py` already encodes both rules** — it resolves the interpreters
once, then loops backends on the outer level calling `build_wheel.py` once per
backend with all interpreters. Prefer `python crates/m3-core-py/build_local.py all`
over hand-rolled loops; it is the same optimization in one command (and adds
uv-based interpreter discovery that rejects the project `.venv`, plus the Linux
zig/native split — see §3a).

> **Why per-version wheels at all?** PyO3 here uses
> `features = ["extension-module"]` *without* `abi3`, so each wheel links a
> specific CPython ABI (the `cpXY` tag) and is **not** cross-version. If the crate
> ever adopted the stable ABI (`abi3-py311`), one wheel would cover 3.11+ — at a
> small perf/feature cost — and this matrix would collapse to one wheel per
> (os, backend). It does not today, so build all four.

### Common prerequisites (all platforms, all backends)

- **Rust ≥ 1.94** (`rustup`), **maturin ≥ 1.7,<2** (`pipx install maturin`)
- A **C/C++ compiler** + **cmake** + **git** (GPU builds cmake-compile llama.cpp)
- **patchelf** (Linux) — maturin needs it to bundle external `.so`s into the
  wheel. Without it the build fails at the repair step:
  `Failed to execute 'patchelf'`.

### 3a. Linux

CPU and the GPU backends differ only in toolchain. Build host: a provisioned
Debian 13 LXC build container (all three backends; see §6 for the login-shell SSH
and CUDA `--no-smoke-test` rules). Any equivalently-toolchained x86_64 Linux box
also works.

**Preferred: one command for the whole matrix.** `build_local.py` applies the
"Optimal build order & caching" rules above automatically — interpreters resolved
once, backends looped on the outer level (so llama.cpp compiles once per backend,
not once per (backend, version)):

```bash
# Builds every backend valid on this host (linux: cpu, vulkan, cuda) × the
# default cp311–314, in the cache-optimal order, smoke-testing each.
python crates/m3-core-py/build_local.py all
# Or a subset / specific Pythons:
python crates/m3-core-py/build_local.py cpu vulkan --pythons 3.11 3.12
```

`build_local.py` handles two Linux-specific details for you:
- **CPU uses maturin `--zig`** for a portable, low-glibc-floor manylinux tag.
- **GPU backends (vulkan/cuda) build NATIVE** — under `--zig`, llama-cpp-sys's
  cmake can't find the host Vulkan/CUDA libs in zig's isolated sysroot
  (`Could NOT find Vulkan (missing: Vulkan_LIBRARY)`), so they must link host
  libs natively. (zig is never used on Windows/macOS.)

**Explicit fallback** — the raw per-backend calls `build_local.py` wraps. Keep the
backend-outer order; pass all interpreters in one call per backend (§ Optimal
build order):

```bash
# CPU — CPU-only llama.cpp embedder, no GPU toolchain beyond cmake + a C/C++
# compiler. Broad manylinux compat (add maturin's --zig for the lowest glibc floor).
python crates/m3-core-py/build_wheel.py --backend cpu --os linux \
    --out dist/m3-core-rs-linux-cpu \
    -- --interpreter python3.11 python3.12 python3.13 python3.14

# Vulkan — needs the shader toolchain + Vulkan dev headers:
#   apt install glslc glslang-tools libvulkan-dev spirv-tools mesa-vulkan-drivers
python crates/m3-core-py/build_wheel.py --backend vulkan --os linux \
    --out dist/m3-core-rs-linux-vulkan \
    -- --interpreter python3.11 python3.12 python3.13 python3.14

# CUDA — needs the CUDA toolkit (nvcc + cuBLAS headers). Not yet built/verified
# locally; CI's Jimver/cuda-toolkit action provides nvcc on the runner.
```

Notes:
- The CPU wheel comes out `manylinux_2_34`; the Vulkan wheel `manylinux_2_38`
  (it links a newer `libvulkan`). Both are broadly installable.
- **glibc floor matters**: build on an *old enough* glibc for your target
  audience. Debian 13 (glibc 2.41) yields `manylinux_2_3x`. For maximum reach,
  build CPU wheels in the official `quay.io/pypa/manylinux_2_28` container.
- **CPU now ships an in-process embedder too** (`--features embedded`), so the
  Linux CPU build also cmake-compiles llama.cpp — the caching rules apply equally
  to all three Linux backends.

### 3b. Windows

Build host: a Windows machine with Visual Studio Build Tools.

**Preferred:** `python crates\m3-core-py\build_local.py all` — same cache-optimal
driver as Linux (one llama.cpp compile per backend). Windows uses native builds
for every backend (no zig). It must run inside a `vcvars64` environment with the
MSVC linker ahead of Git's `link.exe` and `CMAKE_GENERATOR=Ninja` — see the
Windows gotchas in §7. The explicit per-backend `build_wheel.py` calls below are
the fallback:

```powershell
# CPU
python crates\m3-core-py\build_wheel.py --backend cpu --os windows `
    --out dist\m3-core-rs-windows-cpu `
    -- --interpreter python3.11 python3.12 python3.13 python3.14

# CUDA — needs the CUDA Toolkit (nvcc + cuBLAS). Wheel links cublasLt64_*/cublas64_*;
#        maturin warns these CUDA DLLs are NOT bundled (the package __init__.py
#        registers the CUDA dir via os.add_dll_directory at import — CUDA must be
#        installed on the user's box).
# Vulkan — needs the Vulkan SDK + glslc.
```

> **Windows GPU gotcha (documented in PUBLISHING.md):** the Visual Studio CMake
> generator hits a CMake-4.x + MSBuild `ExternalProject` batch-label bug
> (`VCEnd`) while building `vulkan-shaders-gen`. **Force Ninja**: set
> `CMAKE_GENERATOR=Ninja` inside a `vcvars64` environment with `glslc` on PATH.
> CI does this automatically for every GPU build.

### 3c. macOS (Apple Silicon — Metal)

macOS is **Metal-only** by design (Apple Silicon always has a Metal GPU; a
CPU-only mac package would be pointless). Build host: an Apple-Silicon Mac
(this leg is authored in CI but **not yet verified on a device**; verify on a
real Mac before relying on it).

```bash
brew install rustup-init cmake
rustup-init -y && source "$HOME/.cargo/env"
pipx install maturin
python crates/m3-core-py/build_wheel.py --backend metal --os macos \
    --out dist/m3-core-rs-macos-metal \
    -- --interpreter python3.11 python3.12 python3.13 python3.14
```

See `MACOS_BUILD_CONTRIBUTION.md` and `macos-wheels-workflow.yml.template`.

---

## 4. Verifying a wheel

**Always verify off-CI before publishing.** CI only import-smokes CPU wheels
(runners have no GPU device). GPU wheels must be verified on an actual GPU host.

```bash
python3.12 -m venv /tmp/v && /tmp/v/bin/python -m pip install <the>.whl

# 1. Project name resolves the way the wizard installs it:
/tmp/v/bin/python -m pip show m3-core-rs-linux-vulkan | grep -E '^Name|^Version'
#   Name: m3-core-rs-linux-vulkan   Version: 3.7.25

# 2. Imports as m3_core_rs:
/tmp/v/bin/python -c "import m3_core_rs as m; print(m.embed_backend_label())"

# 3. GPU backends only — real embedding + GPU offload (needs a BGE-M3 GGUF):
/tmp/v/bin/python - <<'PY'
import m3_core_rs as m
e = m.EmbeddedEmbedder("/path/to/bge-m3-Q4_K_M.gguf")
v = e.embed(["hello world"])
print("dim", len(v[0]), "backend", m.embed_backend_label())
PY
# With GGML_LOG_LEVEL=info you should see, for a working GPU wheel:
#   ggml_vulkan: Found 1 Vulkan devices: 0 = AMD Radeon Graphics (RADV ...)
#   load_tensors: offloaded N/N layers to GPU
```

Expected for a healthy wheel (GPU **and** CPU): `EmbeddedEmbedder` present,
`embedding_dim` == model dim (BGE-M3 = **1024**), L2 norm ≈ 1.0. `backend_label`
== the backend for GPU wheels, and `cpu` for the CPU wheel. A wheel **without**
`EmbeddedEmbedder` is broken regardless of backend (CPU included — it now builds
`--features embedded`).

The BGE-M3 Q4_K_M GGUF (~418 MB) is an LM Studio asset; if you use LM Studio it
lives under `~/.lmstudio/models/.../bge-m3-GGUF-Q4_K_M.gguf`.

---

## 5. Deployment

There are two channels. **PyPI is primary** (it's what the m3 wizard installs
from); **GitHub Releases** are an interim/secondary mirror.

### 5a. PyPI (canonical) — via CI + Trusted Publishing

1. Bump `workspace.package.version` in the top-level `Cargo.toml` **and**
   `M3_CORE_RS_VERSION` / `M3_CORE_RS_GIT_TAG` in m3-memory's
   `m3_memory/rust_core_install.py`, in lockstep. Current: `3.7.28` ↔ `v2026.7.28`.

   > **CUDA does not publish to PyPI.** Its wheels (~949 MiB linux, ~244 MiB
   > windows) are roughly 10x the 100 MB per-file limit, so `release.yml`'s
   > publish matrix covers the 5 PyPI-eligible backends only and CUDA ships via
   > the GitHub Release. Size is a barrier, not a hurdle — do not re-add CUDA to
   > that matrix expecting a retry to succeed.
2. Ensure each target project has a **trusted publisher** registered on PyPI
   (owner `skynetcmd`, repo `m3-core-rs`, workflow `release.yml`, environment
   `pypi-<os>-<backend>`). See `docs/PUBLISHING.md`.
3. Trigger `release.yml`: push tag `v*`, **or** `gh workflow run release.yml -f
   publish=true`. Use `publish=false` to build-and-artifact only (no upload).
4. The wizard then resolves `pip install --only-binary=:all:
   m3-core-rs-<os>-<backend>==<ver>` for the user's host.

> **No API tokens.** Publishing is OIDC Trusted Publishing. Never add a PyPI
> token to the repo or upload manually with one — it bypasses the design.

#### The pending-publisher rotation (important)

PyPI caps **pending** trusted publishers at **3 at a time** (a name only becomes
a real project after its first publish). With 7 packages you can't pre-register
all of them. The rotation:

1. Register 3 pending publishers (we started with the **Windows** three:
   `pypi-windows-cpu`, `-windows-cuda`, `-windows-vulkan`).
2. Run `release.yml` → those 3 publish → they become real projects → their
   pending slots free.
3. Register the next 3 (the **Linux** three) → publish.
4. Register the last one (`macos-metal`) → publish.

**Do not "re-purpose" a registered publisher** — each is locked to one project
name. Just publish with it; that frees the slot for the next.

### 5b. GitHub Releases (interim mirror)

For users who can't reach PyPI yet (e.g. before all publishers are set up),
attach the verified wheels to the `v<ver>` release as assets. Build a draft
first; don't touch a published release's assets without intent:

> **CI does this for you now.** `release.yml`'s `publish-github-release`
> job attaches EVERY backend's wheels to the Release for the tag
> automatically, so the manual commands below are only for a hand-built
> or backfilled release. They are also the guaranteed path for CUDA,
> whose wheels exceed PyPI's per-file limit (~970 MB linux-cuda) and are
> expected to be rejected there.

```bash
gh release create v2026.7.25 --draft --title "..." --notes "..." \
    dist/m3-core-rs-linux-cpu/*.whl dist/m3-core-rs-linux-vulkan/*.whl
# or attach to an existing draft:
gh release upload v2026.7.25 <wheel>... --clobber
```

Install from a Release asset URL:
`pip install https://github.com/skynetcmd/m3-core-rs/releases/download/v2026.7.25/<wheel>`

> Wheels are build outputs, **not** source — they're `.gitignore`d in both
> repos (`*.whl`, `dist/`). Never `git add` a wheel.

---

## 6. Build hosts (this project's machines)

| Platform | Host | Notes |
|----------|------|-------|
| Linux x86_64 | A provisioned Debian 13 LXC build container | rustc 1.95, maturin 1.13.3, uv, CUDA 13.2 toolkit, Vulkan dev stack + AMD/RADV iGPU. Builds **all three** Linux backends. **Not** the bare Proxmox hypervisor. |
| Windows | A Windows build box | VS Build Tools, CUDA, Vulkan SDK. |
| macOS (Metal) | An Apple-Silicon Mac | Metal build leg to be verified on a device. |

### Linux build box (provisioned Debian 13 LXC)

Two operational rules that bite if missed when building inside an LXC over SSH:

- **Always use a LOGIN shell.** rust / maturin / uv / nvcc are typically only on
  `PATH` in a login shell. Run `ssh <host> 'bash -lc "..."'` — a non-login
  `ssh <host> <cmd>` reports command-not-found. (Fallback if SSH breaks: reach the
  container from the Proxmox host with `pct exec <ctid> -- su - <user> -c "..."`.)
- **GPU verify split depends on the box's GPU.** A build box with an **AMD/RADV
  iGPU and no NVIDIA** can build *and* smoke-test CPU + Vulkan, but **CUDA is
  build-only** there (nvcc cross-compiles without a GPU) — build with
  `--no-smoke-test` and import-verify on an NVIDIA box. Budget ~27 min for a
  from-scratch Linux CUDA build on a 4-core box.

Build the Linux matrix over SSH:

```bash
ssh <host> 'bash -lc "cd ~/m3-core-rs/crates/m3-core-py && python3 build_local.py cpu vulkan"'
ssh <host> 'bash -lc "cd ~/m3-core-rs/crates/m3-core-py && python3 build_local.py cuda --no-smoke-test"'
```

### Linux Vulkan GPU verification (one-time host setup)

If you build/verify GPU wheels inside an unprivileged LXC on Proxmox, the GPU
needs explicit passthrough to *run* (not build) GPU wheels. Example for an AMD
RADV iGPU:

1. **Host**: `apt install mesa-vulkan-drivers vulkan-tools`
   → `vulkaninfo` shows the device (e.g. `AMD Radeon Graphics (RADV ...)`).
2. **CT config** (PVE 9, modern syntax):
   `pct set <ctid> --dev0 /dev/dri/renderD128,gid=993 --dev1 /dev/dri/card0,gid=44`
   then `pct reboot <ctid>`.
3. Inside the CT, GID 993 maps to a group named `kvm` (harmless name collision);
   `usermod -aG kvm <user>`, re-login. `vulkaninfo` inside the CT then sees the GPU.

The **XDNA NPU is not usable** — llama.cpp has no NPU backend; `embedded-vulkan`
targets the GPU only.

### Building all three OSes in parallel

The three OS build hosts are fully independent machines with separate source
checkouts and no shared state, so the whole 28-wheel matrix can be built
**concurrently** — one host per OS — instead of serially. Wall-clock drops from
the *sum* of the three OS builds to the *slowest single* one (almost always the
Linux CUDA leg).

Pattern (one worker per OS, each end-to-end):

1. **Deliver source** to each host at the exact release commit (see the mtime
   caveat in §7 — prefer `git fetch` + `git reset --hard` over `git archive` so
   the CUDA cache stays warm).
2. **Delete the previous build's wheels** for that OS first
   (`ci-wheels/local-<ver>/<os>-*`), so a partial/stale wheel can't masquerade
   as fresh output.
3. **Build** that OS's backends via `build_local.py` (Windows/Linux: `cpu vulkan
   cuda`; macOS: `metal`). GPU/long builds run detached; poll the log + the
   `ci-wheels` dir for completion rather than blocking.
4. **Verify** per OS: import-test where the host can load the wheel (matching
   GPU + interpreter), symbol-scan the compiled extension otherwise (§7).
5. **Collect** all wheels to one box only *after* every OS reports success.

No two workers contend: each writes only its own OS's wheels (distinct package
names), and the final collect step is single-writer. A failure on one host is
isolated — fix and re-run that OS; the other two OSes' wheels still stand. If a
host is unreachable (e.g. a laptop asleep), build the reachable OSes now and run
the missing one as a follow-up wave; the GitHub release draft can be topped up
when it returns.

---

## 7. Gotchas (learned the hard way)

- **A failed build POISONS the next one via `CMakeCache.txt`.** cmake writes the
  resolved compiler flags into its cache, and reuses them on every subsequent
  configure — so a build that failed because of a bad flag keeps failing with the
  *identical* error after you fix the flag. The obvious conclusion ("my fix
  didn't work") is wrong. This cost real time on 2026-07-26. Fix: wipe the
  nested cmake tree, not just `cargo clean`:
  `rm -rf target/release/build/llama-cpp-sys-2-*/out/build`. Verify with
  `grep <the-flag> target/release/build/llama-cpp-sys-2-*/out/build/CMakeCache.txt`
  — an empty result means the cache is genuinely clean.

- **Windows Vulkan: `fatal error C1083: Cannot open compiler generated file: ''`.**
  Reads like a broken MSVC install; it is actually cmake's 250-char
  `CMAKE_OBJECT_PATH_MAX` being exceeded. llama.cpp builds `vulkan-shaders-gen`
  as a nested `ExternalProject`, whose TryCompile lands ~204 chars below the
  cargo target dir, so a normal user clone path overflows. cmake says so — but
  in a warning ~15 lines *above* the error. Two traps: (a) the cap covers the
  directory PLUS the object filename (~12 more), so "under 250" can still fail
  (observed failing at 244); (b) raising `CMAKE_OBJECT_PATH_MAX` does NOT help,
  because `ExternalProject_Add` forwards only the explicit `CMAKE_ARGS` list and
  that variable is not in it. Fix: shorten the path —
  `CARGO_TARGET_DIR=C:\m3t`. `build_local.py` now does this automatically when
  the default would be tight.

- **Stale 0-byte `.so` after a failed build.** A failed maturin run can leave an
  empty `target/release/libm3_core_rs.so`; cargo then sees it as fresh and skips
  the relink (`0.09s Finished`), and patchelf fails `missing ELF header`. Fix:
  `rm target/release/libm3_core_rs.so target/release/deps/libm3_core_rs*.so`
  and the `target/release/.fingerprint/m3-core-py-*` dir, then rebuild.
- **22-byte stub wheel.** A failed wheel-repair step leaves a ~22-byte `.whl`
  stub that "installs" but is empty. **Always copy good wheels off the build box
  immediately** and re-verify size (CPU ≈ 2.8 MB, GPU ≈ 16 MB).
- **CPU/Vulkan same filename collision.** Both produce
  `m3_core_rs_<os>_<backend>-...` only because the project name differs; if you
  build two backends into the *same* `--out` dir they won't collide (names
  differ), but building a backend twice or mixing a bare `maturin build` will.
  Use a per-project `--out` dir.
- **patchelf missing** → `Failed to execute 'patchelf'`. `apt install patchelf`.
- **Determinism.** Embeddings are deterministic across Python versions and
  CPU/GPU — the same text yields identical vectors. A mismatch means a broken
  build.
- **A pure-Rust change rebuilds GPU wheels in ~1-2 min, not from scratch —
  *only if the build directory's file mtimes are preserved*.** If a release only
  adds/changes Rust crates that do **not** touch `llama-cpp-sys` (e.g. new
  pure-logic crates + PyO3 bindings), cargo reuses the cached llama.cpp/ggml
  compile and only relinks the small Rust cdylib (~1-2 min instead of ~27).
  **CAVEAT (learned the hard way):** cargo's freshness check is **mtime-based**,
  so *how you deliver the source to a remote build host matters*:
  - **Local git checkout** (e.g. Windows on the dev box) — mtimes intact, cache
    warm, fast relink. ✅
  - **`git fetch` + `git reset --hard <commit>`** on the box — git only rewrites
    files that actually changed, so the unchanged llama.cpp vendored sources keep
    their mtimes → cache warm. ✅ **Prefer this for remote delivery.**
  - **`git archive <commit> | ssh tar -x`** — tar **resets every file's mtime to
    extract time**, which invalidates cargo's `llama-cpp-sys` fingerprint and
    forces a **full from-scratch CUDA kernel recompile (~12-27 min)** even though
    llama.cpp content is unchanged. ❌ CPU + Vulkan are unaffected (no kernel
    compile); only the CUDA backend pays this. If you must use `git archive`,
    either accept the CUDA rebuild, or `touch -r` the vendored llama.cpp files
    back to their pre-extract mtime, or delete only
    `target/release/.fingerprint/m3-core-py-*` (not the whole `llama-cpp-sys`
    build dir) so cargo relinks the cdylib without recompiling kernels.
- **Verify a wheel's exported symbols WITHOUT importing it** when the build box
  lacks the matching GPU (Linux CUDA on an AMD box) or interpreter (a cp313
  wheel on a cp314 host). A wheel is a zip; the compiled extension is a `.so`
  (Linux) or `.pyd` (Windows). Extract it with `zipfile` and scan the bytes for
  the expected symbol names — no `strings` needed (Windows has none):
  ```python
  import re, sys, zipfile
  with zipfile.ZipFile(sys.argv[1]) as z:
      ext = next(n for n in z.namelist() if n.endswith((".so", ".pyd")))
      blob = z.read(ext)
  text = b" ".join(re.findall(rb"[\x20-\x7e]{4,}", blob)).decode("ascii", "ignore")
  for sym in ("Governor", "fs_walk", "hash_files"):   # the release's new symbols
      print(sym, sym in text)
  ```
  Import-test (`embed_backend_label()` + a real call) on a box that CAN load the
  wheel; symbol-scan elsewhere. Both together cover the matrix.
- **Windows: Git Bash's `link.exe` shadows MSVC's linker.** If you launch the
  build from (or with PATH inherited from) Git Bash / MSYS, Rust may invoke
  `C:\Program Files\Git\usr\bin\link.exe` (GNU coreutils `link`) instead of
  MSVC `link.exe`. The failure is unmistakable:
  ```
  error: linking with `link.exe` failed: exit code: 1
    = note: "C:\\Program Files\\Git\\usr\\bin\\link.exe" ...
    = note: /usr/bin/link: extra operand '...build_script_build...rcgu.o'
            Try '/usr/bin/link --help' for more information.
  ```
  It is **not** the `vulkan-shaders-gen` / `VCEnd` CMake bug — it fails far
  earlier, while linking pyo3 build scripts. Fix: after `vcvars64`, prepend the
  MSVC tools bin dir so its `link.exe` wins:
  ```
  PATH = <VS>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64 ; <ninja dir> ; %PATH%
  ```
  Confirm with `where link` — MSVC's path must print first. `vcvars64` alone is
  not enough when a Git-Bash PATH is inherited into the cmd session.

### Worked example — local Windows + CUDA build (verified 2026-06-22)

Built `m3_core_rs_windows_cuda-3.6.6-cp314` locally end-to-end. Recipe that
worked (PowerShell driving one `cmd.exe` session so the `vcvars64` env persists
into the build):

```powershell
$vcvars   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$ninjaDir = '<python user scripts dir with ninja.exe>'   # pip install ninja puts it here
$linkDir  = '<VS>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64'    # MSVC link.exe — must beat Git's
$cmd = "call `"$vcvars`" && set `"PATH=$linkDir;$ninjaDir;%PATH%`" && set CMAKE_GENERATOR=Ninja " +
       "&& where link && python crates\m3-core-py\build_wheel.py --backend cuda --os windows --release " +
       "--out dist\m3-core-rs-windows-cuda -- --interpreter python3.14 & echo BUILD_EXIT=%ERRORLEVEL%"
cmd.exe /c $cmd
```

Notes that save time on a re-run:
- **`ninja` via `pip install ninja`** lands at the Python *user* Scripts dir
  (`...\AppData\Roaming\Python\PythonXY\Scripts\ninja.exe`), **not** on PATH and
  **not** under `site-packages\ninja\data\bin`. `python -c "import ninja; print(ninja.BIN_DIR)"`
  prints the real dir.
- **`build_wheel.py --out` is relative to the script's own cwd**, so the wheel
  lands at `crates/m3-core-py/dist/m3-core-rs-windows-cuda/…whl`, not repo-root
  `dist/`. Don't hunt for it at the path you passed.
- **CUDA wheel size ≈ 122 MB** here (links CUDA llama.cpp). A few-KB or 22-byte
  result is a stub — rebuild.
- After a failed link, clean stale pyo3 build dirs before retrying:
  `target\release\build\pyo3-*` (a half-linked `.o` makes cargo think it's fresh).
- **Verify after install**, don't assume: `oxidation_probe` must report
  **8/8 native paths present (current)**, `embed_backend_label()` == the backend
  you built (`cuda`/`vulkan`/`metal`, or `cpu` for the CPU wheel), and
  `EmbeddedEmbedder` present for **every** backend including CPU (see §1 — CPU now
  builds `--features embedded`).
- A `--force-reinstall --no-deps` install of the freshly built CUDA wheel over a
  prior one preserves the GPU embedder; it does **not** silently downgrade to a
  CPU/embed-server path.
- **`CMAKE_GENERATOR` trailing space.** Setting it in cmd as
  `set CMAKE_GENERATOR=Ninja && ...` captures the space *before* `&&` into the
  value, so cmake gets `"Ninja "` and fails its exact-match lookup:
  `CMake Error: Could not create named generator Ninja`. Always quote:
  `set "CMAKE_GENERATOR=Ninja"`. (The GPU builds tolerated the stray space; the
  CPU `embedded` llama.cpp build did not — so this surfaced only when building CPU.)
- **Windows SDK `rc.exe` / `mt.exe` must be on PATH** for the CPU `embedded`
  build. CMake's `CMakeTestCCompiler` try-compile links a manifest, and without
  the SDK bin it fails with `--mt=CMAKE_MT-NOTFOUND` and
  `RC Pass 1: command "rc ..." failed: no such file or directory` →
  "The C compiler is not able to compile a simple test program." Fix: prepend
  `C:\Program Files (x86)\Windows Kits\10\bin\<sdk-ver>\x64` (holds `rc.exe`,
  `mt.exe`) to PATH alongside the MSVC link dir. `vcvars64` usually adds it, but
  a Git-Bash-inherited PATH can bury it.

---

## 8. Release history

> Dated snapshots, newest last. These record what was true AT THE TIME —
> the version numbers in them are deliberately NOT updated on a bump, or
> the history stops being history. For the current pinned version see
> §5a and m3-memory's `rust_core_install.py`.

### Snapshot 2026-05-31

- ✅ **Linux CPU + Vulkan**, cp311–cp314 (8 wheels): built via `build_wheel.py`,
  verified (Vulkan does real BGE-M3 GPU offload on the iGPU); attached to the
  `v2026.05.30` Release draft.
- ✅ **Windows CPU/CUDA/Vulkan**: published to the `v2026.05.30` GitHub Release —
  but as **cp314-only** and with the non-canonical `+backend` local-tag name.
  **Re-build** via the fixed `release.yml` (multi-Python + correct names).
- ⏳ **Linux CUDA**: not built (needs nvcc; CI provides it).
- ⏳ **macOS Metal**: workflow authored, **never run on a device** — verify on
  an Apple-Silicon Mac.
- **Trusted publishers**: 3 registered (Windows). Rotate per §5a after first publish.
- **`release.yml`**: fixed this session to build all of 3.11–3.14 (was 3.14-only).

### Update 2026-06-22 — Windows 3.6.6 full matrix (cp311–314)

All three Windows backends built locally via `build_wheel.py` at crate **3.6.6**
with canonical `m3-core-rs-windows-<backend>` names, for the **full Python matrix
cp311 / cp312 / cp313 / cp314** — **12 wheels** total. Built in the cache-optimal
order (one call per backend, all four interpreters; see "Optimal build order"
above), so each backend paid the llama.cpp compile once.

| Backend | Wheel size | `oxidation_probe` | `embed_backend_label()` | `EmbeddedEmbedder` | Pythons |
|---|---|---|---|---|---|
| CPU (`embedded`) | ~2.4 MB | 8/8 functions | `cpu` | ✅ in-process BGE-M3 (dim 1024, L2 1.0) | cp311–314 |
| CUDA | ~122 MB | 8/8 (current) | `cuda` | ✅ | cp311–314 |
| Vulkan | ~20 MB | 8/8 functions | `vulkan` | ✅ | cp311–314 |

Verified in isolated venvs: the cp314 CUDA wheel is installed in the working env
(fixed a stale 3/8 wheel — see §7); the cp312 CPU wheel was loaded in a clean 3.12
venv and embedded BGE-M3 in-process (dim 1024, L2 1.0, `backend_label`=`cpu`).
Wheels are `.gitignore`d — not committed; attach to a Release or publish via CI.
macOS Metal and Linux (all backends) still pending per above.

**Policy change (this build):** the CPU backend now compiles with `--features
embedded` so it ships an in-process BGE-M3 embedder like the GPU wheels — m3
requires a default in-process embedder on *every* build (see §1). The earlier
~0.97 MB embedder-less CPU wheel is superseded. CI/build implication: CPU builds
now need cmake + a C/C++ compiler (previously toolchain-free). The first CPU build
also hit two gotchas now in §7: `CMAKE_GENERATOR=Ninja` must be set **without a
trailing space** (`set "CMAKE_GENERATOR=Ninja"`, not `set CMAKE_GENERATOR=Ninja &&`),
and the Windows SDK `bin\<ver>\x64` (rc.exe / mt.exe) must be on PATH for CMake's
compiler test.
