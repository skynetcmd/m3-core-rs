# m3-core-rs — Wheel Build & Deployment Guide

How to build, verify, and publish the `m3-core-rs` Python wheels across
**Linux, Windows, and macOS**. This is the operational companion to
[`docs/PUBLISHING.md`](docs/PUBLISHING.md) (which covers the PyPI / trusted-
publisher side) and [`crates/m3-core-py/build_wheel.py`](crates/m3-core-py/build_wheel.py)
(the single source of truth for the package-name + feature mapping).

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
| Linux | CPU | `m3-core-rs-linux-cpu` | *(none)* | ❌ no — plain vector ops |
| Linux | CUDA | `m3-core-rs-linux-cuda` | `embedded-cuda` | ✅ yes (NVIDIA) |
| Linux | Vulkan | `m3-core-rs-linux-vulkan` | `embedded-vulkan` | ✅ yes (any Vulkan GPU) |
| Windows | CPU | `m3-core-rs-windows-cpu` | *(none)* | ❌ no |
| Windows | CUDA | `m3-core-rs-windows-cuda` | `embedded-cuda` | ✅ yes |
| Windows | Vulkan | `m3-core-rs-windows-vulkan` | `embedded-vulkan` | ✅ yes |
| macOS (Apple Silicon) | Metal | `m3-core-rs-macos-metal` | `embedded-metal` | ✅ yes |

### Two things that trip people up

1. **CPU = no `EmbeddedEmbedder`, by design.** The plain CPU build links *no*
   llama.cpp (`default = []` in `crates/m3-core-py/Cargo.toml`), so it has no
   in-process embedder. CPU hosts embed via the embed-server path; only the GPU
   backends (`embedded-*`) compile llama.cpp in. `m3_memory/rust_core_install.py`
   mirrors this exactly: `_BACKEND_FEATURES["cpu"] = []`. **A CPU wheel that has
   `EmbeddedEmbedder` was built wrong** (with `--features embedded`).

2. **Python versions are NOT separate projects, tags, or publishers.** One
   project + version holds 4 wheels (`...-cp311-...`, `-cp312-`, `-cp313-`,
   `-cp314-`). pip picks the match. You never make a per-Python tag or
   publisher — that fights the ecosystem. The git tag (`v2026.05.30`) stores
   the **package version** (`3.5.30`) only.

### Naming — use `build_wheel.py`, nothing else

`maturin` has no `--name` flag; the wheel name comes from `[project].name`.
`build_wheel.py` temporarily rewrites that to `m3-core-rs-<os>-<backend>` around
the maturin call. **Always build through it.** A wheel built with a bare
`maturin build` is named `m3_core_rs-...` (project `m3_core_rs`) and the m3
wizard's `pip install m3-core-rs-<os>-<backend>==<ver>` will **not** find it.

> ⚠️ **Do NOT** use a `+cpu` / `+vulkan` PEP 440 *local version* label
> (e.g. `m3_core_rs-3.5.30+vulkan-...`). PyPI **rejects local-version wheels on
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

### Common prerequisites (all platforms, all backends)

- **Rust ≥ 1.94** (`rustup`), **maturin ≥ 1.7,<2** (`pipx install maturin`)
- A **C/C++ compiler** + **cmake** + **git** (GPU builds cmake-compile llama.cpp)
- **patchelf** (Linux) — maturin needs it to bundle external `.so`s into the
  wheel. Without it the build fails at the repair step:
  `Failed to execute 'patchelf'`.

### 3a. Linux

CPU and the GPU backends differ only in toolchain. Build host: any x86_64 Linux
box (a Debian 13 container works well — see §6).

```bash
# CPU — plain vector ops, no GPU toolchain needed. Broad manylinux compat.
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

### 3b. Windows

Build host: a Windows machine with Visual Studio Build Tools.

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
#   Name: m3-core-rs-linux-vulkan   Version: 3.5.30

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

Expected for a healthy GPU wheel: `EmbeddedEmbedder` present, `embedding_dim`
== model dim (BGE-M3 = **1024**), L2 norm ≈ 1.0, `backend_label` == the backend.
For a **CPU** wheel: no `EmbeddedEmbedder`, `backend_label` == `none`/`cpu`.

The BGE-M3 Q4_K_M GGUF (~418 MB) is an LM Studio asset; if you use LM Studio it
lives under `~/.lmstudio/models/.../bge-m3-GGUF-Q4_K_M.gguf`.

---

## 5. Deployment

There are two channels. **PyPI is primary** (it's what the m3 wizard installs
from); **GitHub Releases** are an interim/secondary mirror.

### 5a. PyPI (canonical) — via CI + Trusted Publishing

1. Bump `workspace.package.version` in the top-level `Cargo.toml` **and**
   `M3_CORE_RS_VERSION` / `M3_CORE_RS_GIT_TAG` in m3-memory's
   `m3_memory/rust_core_install.py`, in lockstep. Current: `3.5.30` ↔ `v2026.05.30`.
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

```bash
gh release create v2026.05.30 --draft --title "..." --notes "..." \
    dist/m3-core-rs-linux-cpu/*.whl dist/m3-core-rs-linux-vulkan/*.whl
# or attach to an existing draft:
gh release upload v2026.05.30 <wheel>... --clobber
```

Install from a Release asset URL:
`pip install https://github.com/skynetcmd/m3-core-rs/releases/download/v2026.05.30/<wheel>`

> Wheels are build outputs, **not** source — they're `.gitignore`d in both
> repos (`*.whl`, `dist/`). Never `git add` a wheel.

---

## 6. Build hosts (this project's machines)

| Platform | Host | Notes |
|----------|------|-------|
| Linux x86_64 | any x86_64 Linux box | Debian 13, rustup, Python 3.11–3.14. An iGPU/dGPU passed through for Vulkan verify (see below). |
| Windows | a Windows build box | VS Build Tools, CUDA, Vulkan SDK. |
| macOS (Metal) | an Apple-Silicon Mac | Metal build leg to be verified on a device. |

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

---

## 7. Gotchas (learned the hard way)

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
  you built (`cuda`), and `EmbeddedEmbedder` present for GPU backends. A CPU
  build legitimately has neither — see §1.
- A `--force-reinstall --no-deps` install of the freshly built CUDA wheel over a
  prior one preserves the GPU embedder; it does **not** silently downgrade to a
  CPU/embed-server path.

---

## 8. Current state (2026-05-31)

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

### Update 2026-06-22 — Windows 3.6.6 local builds (cp314)

All three Windows backends rebuilt locally via `build_wheel.py` at crate
**3.6.6** with canonical `m3-core-rs-windows-<backend>` names (not the old
`+backend` local tags), cp314, and verified in isolated venvs:

| Backend | Wheel size | `oxidation_probe` | `embed_backend_label()` | `EmbeddedEmbedder` |
|---|---|---|---|---|
| CPU | ~0.97 MB | 8/8 functions | `none` | ❌ (correct — §1) |
| CUDA | ~122 MB | 8/8 (current) | `cuda` | ✅ |
| Vulkan | ~20 MB | 8/8 functions | `vulkan` | ✅ |

The CUDA wheel is installed in the working env (fixed a stale 3/8 wheel — see
§7); CPU and Vulkan were verified in throwaway venvs only. Still cp314-only —
re-run with `--interpreter python3.11 python3.12 python3.13 python3.14` (via `uv`,
§2) for the full matrix before publishing. macOS Metal and Linux CUDA still
pending per above.
