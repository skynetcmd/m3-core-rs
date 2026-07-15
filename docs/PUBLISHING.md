# Publishing m3-core-rs wheels

m3-core-rs is one Rust source tree published to PyPI as **seven** packages,
one per (OS, backend) pair. All install the same `m3_core_rs` import module —
only one is installed at a time. The m3 setup wizard detects the host's OS +
GPU and runs `pip install m3-core-rs-<os>-<backend>` for the matching package.

| OS | Backend | PyPI package |
|----|---------|--------------|
| Windows | CPU | `m3-core-rs-windows-cpu` |
| Windows | CUDA | `m3-core-rs-windows-cuda` |
| Windows | Vulkan | `m3-core-rs-windows-vulkan` |
| Linux | CPU | `m3-core-rs-linux-cpu` |
| Linux | CUDA | `m3-core-rs-linux-cuda` |
| Linux | Vulkan | `m3-core-rs-linux-vulkan` |
| macOS (Apple Silicon) | Metal | `m3-core-rs-macos-metal` |

macOS is Metal-only by design — Apple Silicon always has a Metal GPU, so a
CPU-only mac package would be pointless.

Each (os, backend) ships across the supported CPython matrix: **3.11, 3.12,
3.13, 3.14** (cp311–cp314). A full release is therefore 7 packages × 4
interpreters = 28 wheels.

## Distribution policy: PyPI for wheels that fit, GitHub Release for all of them

Both destinations are used, split by size — they are not alternatives:

- **PyPI** carries every wheel **that fits under PyPI's 100 MB per-file limit**
  — CPU, Vulkan, and Metal (all ≤ ~65 MB). These resolve via
  `pip install m3-core-rs-<os>-<backend>` and are how the `m3 setup` wizard
  installs on most hosts.
- **The GitHub Release** carries **every** wheel — a complete set including the
  CUDA builds, which are far over PyPI's limit (Linux CUDA is ~600 MB–1 GB
  depending on version; see below). It is the guaranteed source of truth for
  the whole matrix, and the only home for CUDA today.

The wizard resolves **PyPI first, then the GitHub Release**, so a host always
finds its wheel regardless of which destination holds it. CUDA is not a
second-class backend — it's the fastest one and ships complete; it lives on the
Release only because a self-contained ~1 GB wheel can't go on PyPI. (A PyPI
file-size-limit increase has been requested upstream; until it lands, CUDA
stays on the Release. The CI workflow re-attempts the PyPI upload every release,
so it starts publishing automatically if the limit rises.)

## Two build workflows

Distribution is by size (above); *building* has two paths.

| Workflow | Builds on | Publishes to | When |
|----------|-----------|--------------|------|
| **Local per-machine** | one box per OS (Windows / Linux / macOS) | GitHub Release via `gh release upload` | routine releases; GPU wheels verified on real hardware |
| CI `release.yml` | GitHub-hosted runners | PyPI (wheels that fit) **and** GitHub Release (all wheels) | Linux CUDA (no local NVIDIA box); the automated tag-push path |

The local flow is preferred for hand-controlled waves and hardware-verified GPU
wheels. The CI flow builds all seven on native runners and, on a `v*` tag,
attempts PyPI for each (CUDA fails non-fatally on size) **and** attaches the
complete set to the Release — so a single tag push produces both destinations.
The two share `build_wheel.py` (the name/feature mapping) and the GPU build
gotchas below.

## Current workflow — build locally on each platform's box

We build each platform's wheels on a real machine of that platform and attach
them to a GitHub **release** (draft until ready), rather than running the
all-seven CI matrix. This sidesteps CI's GPU-build fragility, lets GPU wheels
be smoke-tested on actual hardware, and keeps each wave (CPU → Vulkan → CUDA)
under direct control.

### Build machines

| Platform | Builder | Toolchain prerequisites | Backends built there |
|----------|---------|-------------------------|----------------------|
| Windows x86_64 | a Windows dev box | MSVC + Rust + maturin; CUDA toolkit + nvcc; Vulkan SDK; an NVIDIA GPU to verify CUDA | cpu, vulkan, cuda — all three build **and** verify |
| Linux x86_64 | a Linux build host (we use an LXC container on the Proxmox box) | Rust + maturin + uv; Vulkan dev stack (`libvulkan-dev`, `glslang-tools`, `libshaderc`) | cpu, vulkan — Vulkan verifies if the host exposes a Vulkan device |
| macOS arm64 | an Apple Silicon Mac | Xcode CLT + Rust + maturin | metal |

(Operator-specific hostnames, IPs, and SSH aliases for our build machines are
kept out of this public doc — see the private operator notes.)

Notes that bite if forgotten:

- **Linux CUDA needs an NVIDIA GPU + CUDA toolkit.** If your Linux build host
  has no NVIDIA GPU (e.g. an AMD/integrated-GPU box), **build Linux CUDA via CI
  instead** (see the CI workflow section) — that box can still build and verify Vulkan
  on its own GPU.
- **Login shell on the Linux box:** if rust/maturin/uv were installed per-user
  (cargo/uv under `~`), they are only on `PATH` in a **login** shell — run
  remote builds with `ssh <host> 'bash -lc "..."'`, not a bare `ssh <host> cmd`.

### Per-platform build command

One cross-platform driver, `crates/m3-core-py/build_local.py`, handles all
three OSes. It detects the host OS, resolves the four CPython interpreters via
`uv` (preferring clean uv-managed installs and refusing project virtualenvs),
applies the per-OS build rules, then calls `build_wheel.py` once per backend
and smoke-tests each result. Run it from anywhere in the repo:

```bash
# from crates/m3-core-py/  (use a LOGIN shell on Linux so uv/maturin are on PATH)
python build_local.py cpu                  # one backend
python build_local.py cpu vulkan cuda      # several
python build_local.py all                  # every backend valid for this OS
python build_local.py vulkan --no-smoke-test
python build_local.py cpu --pythons 3.12 3.13   # subset of interpreters
```

Output lands in `ci-wheels/local-<ver>/<os>-<backend>/*.whl` with a sibling
`build-<os>-<backend>.log`. Tags: `win_amd64` (Windows), `macosx_11_0_arm64`
(macOS), `manylinux*` (Linux).

**Per-OS rules the driver encodes for you:**

- **Linux CPU uses `maturin --zig`** for a portable `manylinux2014` tag (no
  Docker). One-time host setup: `pipx inject maturin ziglang` **plus** a `zig`
  binary shim at `~/.local/bin/zig` that execs `python -m ziglang "$@"`
  (maturin needs a `zig` executable on PATH, not just the module).
- **Linux GPU backends build NATIVE (no `--zig`).** Under `--zig`,
  llama-cpp-sys's cmake step can't find the system Vulkan/CUDA library in zig's
  isolated sysroot (`Could NOT find Vulkan (missing: Vulkan_LIBRARY)`).
  `build_local.py` applies `--zig` only for Linux CPU; everything else links
  the host libs natively (tags like `manylinux_2_38`, matching the host glibc).
  Linux GPU builds also need `rustup component add rustfmt` (llama-cpp-sys
  bindgen wants it).
- **Windows GPU** needs `CMAKE_GENERATOR=Ninja` inside a `vcvars64` shell with
  `glslc` on PATH (see the GPU-build note above); run `build_local.py cuda` /
  `vulkan` from a Developer Command Prompt.
```

### Smoke test before upload

Install each built wheel into a throwaway env and confirm the backend label:
```bash
uv venv --python 3.13 /tmp/v && uv pip install --python /tmp/v/bin/python <wheel>
/tmp/v/bin/python -c "import m3_core_rs; print(m3_core_rs.embed_backend_label())"
# expect: none (cpu) | vulkan | cuda | metal
```

### Publish to a GitHub Release

Collect every platform's wheels onto one machine (e.g. `scp` the Linux/mac
wheels to the Windows box) and attach them to the release tag with `--clobber` (idempotent
— safe to re-run):
```bash
gh release upload v2026.06.07 <wheel> [<wheel> ...] --clobber
gh release view  v2026.06.07 --json assets --jq '.assets[].name' | sort   # verify
```
The release stays a **draft** until every wave is attached; publish when the
matrix is complete. Tag convention is date-based: `v2026.06.07` for the 3.6.6
release.

## Single source of truth

`crates/m3-core-py/build_wheel.py` owns the `(os, backend) -> (package name,
maturin features)` mapping. Both local builds and CI call it, so the names can
never drift. The m3-memory wizard mirrors the same mapping in
`m3_memory/rust_core_install.py::package_name` — keep the two in sync.

## Building one wheel locally

```bash
# from crates/m3-core-py/
python build_wheel.py --backend cpu                 # OS inferred from host
python build_wheel.py --backend cuda  --os windows  # needs CUDA toolkit + nvcc
python build_wheel.py --backend vulkan --os windows # needs Vulkan SDK; uses Ninja
```

GPU builds compile llama.cpp from source (cmake + a C/C++ compiler + libclang
for bindgen). On Windows the Visual Studio CMake generator hits a CMake-4.x +
MSBuild ExternalProject batch-label bug (`VCEnd`) while building
`vulkan-shaders-gen`; set `CMAKE_GENERATOR=Ninja` (inside a `vcvars64`
environment, with `glslc` on PATH) to avoid it. The CI workflow does this
automatically for every GPU build.

## CI `.github/workflows/release.yml` — the automated tag-push path

> The all-in-CI build: every wheel built on native runners on a `v*` tag push.
> The local per-machine workflow above is preferred for hand-controlled waves
> and hardware-verified GPU wheels, but CI is the path for **Linux CUDA** (the
> local Linux box has no NVIDIA GPU) and a valid fallback for any backend.

Triggered on a `v*` tag push (or `workflow_dispatch` with `publish=false` to
build-only). It builds all seven wheels on native runners, then in two jobs:

- **`publish`** — attempts to upload each package to its PyPI project. The CPU /
  Vulkan / Metal wheels publish normally. The CUDA publish step is
  `continue-on-error` (with `fail-fast: false` on the matrix), so PyPI rejecting
  an over-limit CUDA wheel is a *skipped* job, not a failed run — and if PyPI's
  limit ever rises (a limit increase has been requested upstream), CUDA will
  start publishing automatically with no workflow change.
- **`publish-github-release`** — downloads **all** built wheels and attaches them
  to the tag's GitHub Release, so the Release is always the complete set
  regardless of what PyPI accepted.

So one tag push populates both destinations per the size policy above.

GitHub-hosted runners have **no GPU** — that is fine. CI *builds* the backends
(nvcc compiles CUDA kernels, glslc compiles Vulkan shaders); it never runs
them on a device. Only the CPU wheels are import-smoke-tested in CI.

### Linux CUDA build recipe (CI)

The Linux CUDA toolkit is **fetched on the runner** — there is no local Linux
CUDA toolkit anywhere. To build 3.6.6 Linux CUDA wheels, dispatch `release.yml`
with `publish=false` and pull the `linux-cuda` artifact. The known-good config:

- **Toolkit:** `Jimver/cuda-toolkit@>=v0.2.32` (used `v0.2.35`), CUDA **13.2.0**.
  The shipped 3.5.30 wheel's fatbin spans `sm_75`…`sm_120`/`compute_120`
  (Turing → Blackwell, covers the RTX 5080).
- **PIC fix:** `CMAKE_CUDA_FLAGS="-Xcompiler -fPIC"` +
  `CMAKE_POSITION_INDEPENDENT_CODE=ON` (nvcc `.cu.o` is non-PIC but links into a
  `cdylib`).
- **auditwheel:** `--auditwheel skip` for CUDA — `libcuda.so.1` is the driver
  stub and must **not** be bundled. The resulting wheel has no `.libs/`; the
  CUDA runtime is statically linked into the extension `.so` (tag
  `manylinux_2_39`).

## PyPI Trusted Publishing — one-time human setup (CI PyPI path)

Applies to the CI `release.yml` PyPI-publish job; the local build workflow
attaches wheels to a GitHub Release instead and does not touch PyPI. Publishing
via CI uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no API tokens are stored in the repo. Before the workflow can publish,
**each PyPI project** must register this repo + workflow as a trusted publisher:

For every package `m3-core-rs-<os>-<backend>`:

1. Create the project on PyPI (or do the first upload manually to claim the name).
2. Project → *Settings* → *Publishing* → *Add a trusted publisher* → GitHub:
   - **Owner:** `skynetcmd`
   - **Repository:** `m3-core-rs`
   - **Workflow:** `release.yml`
   - **Environment:** `pypi-<os>-<backend>` (e.g. `pypi-windows-cuda`) —
     must match the `environment.name` in the publish job's matrix.

> **Setup is a two-pass process because PyPI caps *pending* publishers at 3.**
> A pending publisher (registered before the project's first upload) occupies
> one of three slots; once it actually publishes once, it becomes a real project
> and frees its slot. So register the first 3, publish them, then register the
> next batch. A publisher binds `(repo, workflow, environment) → one fixed
> package name` and can't be re-pointed at another backend.
>
> **Current registration state (as of the last setup pass):** the 3 Windows
> publishers (`pypi-windows-{cpu,cuda,vulkan}`) are registered; `linux-{cpu,cuda,
> vulkan}` and `macos-metal` are **not yet** — they're waiting on the Windows 3
> to publish and free the pending slots. Note the CUDA projects still won't
> accept an over-limit wheel even once registered — that's expected; CUDA lives
> on the Release (see the distribution policy).

Until the projects you need are registered, run the workflow with
`publish=false` and grab the wheels from the run's artifacts (the
`publish-github-release` job still attaches the complete set to the Release on a
real tag push).

## Version / tag alignment

The wheel version comes from `workspace.package.version` in the top-level
`Cargo.toml` (maturin reads it via the `dynamic = ["version"]` declaration).
Bump it in lockstep with the release tag. `m3-memory` pins the expected
version in `m3_memory/rust_core_install.py` (`M3_CORE_RS_VERSION` /
`M3_CORE_RS_GIT_TAG`) — bump those together.

Current: `3.6.6` ↔ tag `v2026.06.07` (built via the local per-machine workflow;
release `v2026.06.07` holds Windows cpu/vulkan/cuda, Linux cpu/vulkan, and macOS
metal — Linux CUDA pending via CI).

Prior: `3.5.30` ↔ tag `v2026.05.30` (built via the legacy all-in-CI workflow).
