# Contribution: macOS Wheel Automation & Multi-Python Support

This contribution adds automated build support for macOS (Apple Silicon/Intel) wheels and the `m3-embed-server` binary. It enables high-performance Metal GPU acceleration and "sovereign" CPU-only modes for macOS users.

## Summary of Changes
1. **GitHub Actions Workflow**: Added `.github/workflows/macos-wheels.yml` to automate wheel builds for Python 3.11, 3.12, and 3.14 on every push/PR.
2. **Multi-Python Support**: Verified and documented the process for building ABI-specific wheels using the `--interpreter` flag in `maturin`.
3. **Artifact Renaming Pattern**: Recommended a clear naming convention for the CPU (`-cpu`) and Metal (`-metal`) variants to simplify installation for end-users.

## Manual Build Verification (Local)
The following commands were used to verify the builds locally on macOS Tahoe (arm64):

```bash
# 1. Install Toolchain
brew install rustup-init cmake
rustup-init -y && source $HOME/.cargo/env

# 2. Build for Python 3.12 (Standard)
cd crates/m3-core-py
maturin build --release --features embedded --interpreter python3.12
maturin build --release --features embedded-metal --interpreter python3.12

# 3. Build Embed Server
cargo build -p m3-embed-server --release --features embedded
```

## Installation
```bash
# Install the Metal-optimized wheel
pip install target/wheels/m3_core_rs-metal-cp312.whl
```
