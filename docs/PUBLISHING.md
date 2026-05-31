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

## CI: `.github/workflows/release.yml`

Triggered on a `v*` tag push (or `workflow_dispatch` with `publish=false` to
build-only). It builds all seven wheels on native runners and publishes each
to its PyPI project.

GitHub-hosted runners have **no GPU** — that is fine. CI *builds* the backends
(nvcc compiles CUDA kernels, glslc compiles Vulkan shaders); it never runs
them on a device. Only the CPU wheels are import-smoke-tested in CI.

## PyPI Trusted Publishing — one-time human setup

Publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
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

Current: `3.5.30` ↔ tag `v2026.05.30`.
