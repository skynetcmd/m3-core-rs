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

## Two release workflows

There are two ways these wheels get built and shipped. **The current,
preferred workflow is local per-machine builds** (next section). The older
all-in-CI flow is retained below as legacy reference — it was the original
one-off mechanism and parts of it (the cross-platform `build_wheel.py`
mapping, the GPU build gotchas) are still used by the local flow.

| Workflow | Builds on | Distribution | Status |
|----------|-----------|--------------|--------|
| **Local per-machine** | one box per OS (Windows / Linux / macOS) | GitHub Release assets via `gh release upload` | **Current** |
| CI `release.yml` | GitHub-hosted runners | PyPI Trusted Publishing | Legacy / one-off |

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
  instead** (see legacy section) — that box can still build and verify Vulkan
  on its own GPU.
- **Login shell on the Linux box:** if rust/maturin/uv were installed per-user
  (cargo/uv under `~`), they are only on `PATH` in a **login** shell — run
  remote builds with `ssh <host> 'bash -lc "..."'`, not a bare `ssh <host> cmd`.

### Per-platform build commands

Each platform has a thin wrapper script that resolves the four interpreters and
calls `build_wheel.py` once per backend. All wrappers funnel through the same
`build_wheel.py` single-source-of-truth mapping.

**Windows** — `build_windows_wheels.sh <cpu|vulkan|cuda>`:
```bash
# resolves the four uv-managed CPython 3.11–3.14, runs one maturin build per backend.
bash build_windows_wheels.sh cpu
bash build_windows_wheels.sh vulkan   # needs Vulkan SDK
bash build_windows_wheels.sh cuda     # needs CUDA toolkit + nvcc (run inside vcvars64)
# → ci-wheels/local-<ver>/windows-<backend>/*.whl  (tag: win_amd64)
```

**Linux** — `build_linux_wheels.sh <cpu|vulkan|cuda>`:
```bash
# on the Linux build host (login shell so cargo/maturin/uv are on PATH):
cd ~/m3-core-rs && ./build_linux_wheels.sh cpu
cd ~/m3-core-rs && ./build_linux_wheels.sh vulkan
```
- **CPU uses `maturin --zig`** for a portable `manylinux2014` tag (no Docker).
  One-time setup: `pipx inject maturin ziglang` **plus** a `zig` binary shim at
  `~/.local/bin/zig` that execs `python -m ziglang "$@"` (maturin needs a `zig`
  executable on PATH, not just the module).
- **GPU backends build NATIVE (no `--zig`).** Under `--zig`, llama-cpp-sys's
  cmake step can't find the system Vulkan/CUDA library in zig's isolated
  sysroot (`Could NOT find Vulkan (missing: Vulkan_LIBRARY)`). The wrapper
  applies `--zig` only for `cpu`; vulkan/cuda omit it and link the host's
  libs (tags `manylinux_2_38` etc., matching the host glibc). Also
  `rustup component add rustfmt` (llama-cpp-sys bindgen wants it).

**macOS** — `build_wheel.py` directly (or the mac wrapper):
```bash
# from crates/m3-core-py/, on the Mac
python build_wheel.py --backend metal -- --interpreter python3.11 python3.12 python3.13 python3.14
# → m3_core_rs_macos_metal-<ver>-cpXY-cpXY-macosx_11_0_arm64.whl
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

## Legacy / one-off: CI `.github/workflows/release.yml`

> This is the **original** publishing mechanism — an all-in-CI build of every
> wheel, published straight to PyPI via Trusted Publishing. The local
> per-machine workflow above has superseded it for routine releases. CI is
> still the path for **Linux CUDA** (the local Linux box has no NVIDIA GPU),
> and remains a valid fallback for any backend.

Triggered on a `v*` tag push (or `workflow_dispatch` with `publish=false` to
build-only). It builds all seven wheels on native runners and publishes each
to its PyPI project.

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

## PyPI Trusted Publishing — one-time human setup (legacy CI path)

Applies only to the CI `release.yml` PyPI-publish path above; the local
workflow attaches wheels to a GitHub Release instead and does not touch PyPI.
Publishing via CI uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no API tokens are stored in the repo. Before the workflow can publish,
**each of the seven PyPI projects** must register this repo + workflow as a
trusted publisher:

For every package `m3-core-rs-<os>-<backend>`:

1. Create the project on PyPI (or do the first upload manually to claim the name).
2. Project → *Settings* → *Publishing* → *Add a trusted publisher* → GitHub:
   - **Owner:** `skynetcmd`
   - **Repository:** `m3-core-rs`
   - **Workflow:** `release.yml`
   - **Environment:** `pypi-<os>-<backend>` (e.g. `pypi-windows-cuda`) —
     must match the `environment.name` in the publish job's matrix.

Until all seven are registered, run the workflow with `publish=false` and grab
the wheels from the run's artifacts.

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
