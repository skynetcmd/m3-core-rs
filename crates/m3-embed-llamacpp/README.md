# m3-embed-llamacpp

llama.cpp embedding backend for the m3 dispatcher. Two backends share one
`ModelBackend` API:

- `HttpBackend` — POSTs to a separate `llama-server` process. Default build,
  no llama.cpp link.
- `EmbeddedBackend` — links llama-cpp-rs in-process, zero IPC. Gated behind
  the `embedded` family of cargo features.

## Build matrix

| feature             | backend  | toolchain required             |
|---------------------|----------|--------------------------------|
| (default)           | HTTP     | none                           |
| `embedded`          | CPU      | C++ compiler (already needed)  |
| `embedded-cuda`     | CUDA     | CUDA Toolkit + nvcc            |
| `embedded-vulkan`   | Vulkan   | Vulkan SDK                     |
| `embedded-metal`    | Metal    | macOS + Xcode                  |

GPU backends are mutually exclusive — pick exactly one. Default `embedded`
remains CPU-only. Each GPU feature additively enables `embedded`, so
`--features embedded-cuda` is sufficient (no need to also pass `embedded`).

Mutual exclusivity is enforced at compile time via `compile_error!` in
`src/lib.rs`; enabling two GPU features simultaneously fails the build with
a clear message rather than handing a conflicting set of selectors to the
underlying llama-cpp-sys-2 C build.

The `embedded-metal` feature additionally gates on `target_os = "macos"` —
attempting to build it on Linux/Windows produces a clear compile error rather
than a silent CPU-only build.

The active backend at runtime is reported by `active_backend()` (re-exported
to Python as `m3_core_rs.embed_backend_label()`): one of `"cpu"`,
`"cuda"`, `"vulkan"`, `"metal"`, or `"none"` if the wheel was built without
`embedded`.

## Toolchain requirements per GPU backend

| feature             | required toolchain                   | tested on this repo  |
|---------------------|--------------------------------------|----------------------|
| `embedded`          | C++ compiler (MSVC / clang / gcc)    | yes — Windows/Linux  |
| `embedded-cuda`     | CUDA Toolkit >= 12.0 + nvcc          | yes — Windows RTX 5080 |
| `embedded-vulkan`   | Vulkan SDK >= 1.3 + glslc            | wired, not built     |
| `embedded-metal`    | macOS + Xcode Command Line Tools     | wired, gated to macOS |

### Vulkan setup (Linux / Windows)
1. Install Vulkan SDK from https://www.lunarg.com/vulkan-sdk/ (or your distro's
   package, e.g. `apt install vulkan-sdk libvulkan-dev`).
2. Confirm `$env:VULKAN_SDK` / `$VULKAN_SDK` is set after install (the installer
   usually does this — restart your shell). `llama-cpp-sys-2`'s `build.rs`
   panics at `build.rs:768` with `"Please install Vulkan SDK and ensure that
   VULKAN_SDK env variable is set: NotPresent"` when the env var is missing.
3. Build: `cargo build -p m3-embed-server --release --features embedded-vulkan`.

### Metal setup (macOS only)
1. Install Xcode Command Line Tools: `xcode-select --install`.
2. Build: `cargo build -p m3-embed-server --release --features embedded-metal`.

> **Caveat — `embedded-metal` on non-Apple targets.** llama-cpp-sys-2's
> `build.rs` only emits `framework=Metal/MetalKit` link directives when the
> target OS is Apple. On Linux/Windows, enabling `embedded-metal` builds
> successfully but produces a CPU-only artifact (the `metal` cargo feature is
> a no-op there). Always cross-check `active_backend()` at runtime if you rely
> on Metal acceleration. Do **not** ship a Linux/Windows wheel built with
> `--features embedded-metal` expecting GPU offload.

### Mutual exclusivity
GPU features are mutually exclusive (enforced via `compile_error!` in
`src/lib.rs`). Cargo will let you specify multiple, but the build will fail
with a clear message.
