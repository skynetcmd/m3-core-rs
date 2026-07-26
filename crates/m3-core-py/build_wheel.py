#!/usr/bin/env python3
"""Build a backend-specific m3-core-rs wheel from the single crate source.

m3-core-rs ships one Rust source tree but is published to PyPI as several
*differently named* packages, one per (OS, backend) pair:

    m3-core-rs-windows-cpu      m3-core-rs-linux-cpu
    m3-core-rs-windows-cuda     m3-core-rs-linux-cuda
    m3-core-rs-windows-vulkan   m3-core-rs-linux-vulkan
                                m3-core-rs-macos-metal

All of them install the SAME import module (`m3_core_rs`), so application
code never changes; only one is installed at a time. The m3 setup wizard
detects the user's OS + GPU and runs `pip install m3-core-rs-<os>-<backend>`
for the matching package.

maturin has no `--name` flag — the wheel name comes from `[project].name`
in `pyproject.toml`. This script is the single source of truth for the
(os, backend) -> (package name, maturin features) mapping. It temporarily
rewrites `[project].name`, invokes `maturin build` with the right features,
then restores the original `pyproject.toml`. Both local builds and the CI
release workflow call this same script so the mapping can never drift.

Usage:
    python build_wheel.py --backend cuda [--os windows] [--out DIR] \
        [-- <extra maturin args>]

If --os is omitted it is inferred from the host. The backend must be valid
for the OS (see _MATRIX). Exit code is maturin's exit code.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PYPROJECT = _HERE / "pyproject.toml"
_WORKSPACE = _HERE.parent.parent  # crates/m3-core-py -> repo root
# maturin copies python-source (`python/`) into the wheel, so a file staged at
# python/m3_core_rs/<x> lands at m3_core_rs/<x> in the installed wheel — exactly
# where m3-memory's embedder_admin._server_binary() looks first. This is where
# we drop the m3-embed-server binary so it ships INSIDE the wheel.
_PY_PKG_DIR = _HERE / "python" / "m3_core_rs"
_EMBED_SERVER_BIN = "m3-embed-server"

# The canonical build matrix. Key: (os, backend). Value: maturin Cargo
# features to activate. The package name is derived as
# `m3-core-rs-<os>-<backend>` for every entry; keeping the name implicit
# guarantees it matches what the wizard computes from the same two tokens.
#
# CPU uses `embedded` (CPU-only llama.cpp, no GPU backend) so EVERY build —
# including CPU — ships an in-process BGE-M3 `EmbeddedEmbedder`. m3-memory
# requires a default in-process bge-m3 embedder on all builds; a CPU wheel
# with no embedder forced reliance on the embed-server (tier 2), which is not
# guaranteed present on an offline host. `embedded` cmake-builds llama.cpp, so
# CPU builds now need a C/C++ compiler + cmake (was toolchain-free before).
# CPU wheel grows ~1 MB -> ~2.4 MB; still far below the GPU wheels (20-122 MB).
#
# macOS is Metal-only: Apple Silicon always has a Metal GPU, so a CPU-only
# mac package would be pointless and is intentionally absent.
_MATRIX: dict[tuple[str, str], list[str]] = {
    ("windows", "cpu"): ["embedded"],
    ("windows", "cuda"): ["embedded-cuda"],
    ("windows", "vulkan"): ["embedded-vulkan"],
    ("linux", "cpu"): ["embedded"],
    ("linux", "cuda"): ["embedded-cuda"],
    ("linux", "vulkan"): ["embedded-vulkan"],
    ("macos", "metal"): ["embedded-metal"],
}


def package_name(os_tok: str, backend: str) -> str:
    """The PyPI project name for an (os, backend) pair. Single source of truth.

    Mirrored verbatim by the m3 wizard's resolver — keep the two in sync."""
    return f"m3-core-rs-{os_tok}-{backend}"


def host_os() -> str:
    """Map the host platform to our OS token (windows/linux/macos)."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


@contextlib.contextmanager
def _patched_name(name: str):
    """Temporarily rewrite [project].name in pyproject.toml, then restore.

    Rewrites only the single `name = "..."` line under [project]; the rest of
    the file (dynamic version, [tool.maturin], etc.) is untouched. The original
    bytes are restored in a finally block even if maturin raises, so an
    interrupted build never leaves the tree with a backend name committed.
    """
    original = _PYPROJECT.read_text(encoding="utf-8")
    in_project = False
    out_lines: list[str] = []
    replaced = False
    for line in original.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
        if in_project and not replaced and stripped.startswith("name") and "=" in stripped:
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f'{indent}name = "{name}"\n')
            replaced = True
            continue
        out_lines.append(line)
    if not replaced:
        raise SystemExit("error: could not find [project].name in pyproject.toml")
    try:
        _PYPROJECT.write_text("".join(out_lines), encoding="utf-8")
        yield
    finally:
        _PYPROJECT.write_text(original, encoding="utf-8")


@contextlib.contextmanager
def _staged_embed_server(features: list[str], release: bool):
    """Build the m3-embed-server binary with the SAME backend as this wheel and
    stage it under python/m3_core_rs/ so maturin bundles it into the wheel.

    m3-embed-server is the shared-embedder baseline: every m3 process defers to
    it on :8082 (one model in host RAM instead of N). It is a separate Cargo
    [[bin]] crate, and `maturin build` packages only the Python extension
    (cdylib), NOT bin targets — so historically the binary was silently dropped
    from the wheel and the runtime fell back to the slower path. Building it here
    with the matching feature (a cuda wheel ships a cuda server) and dropping it
    into the python-source dir makes it ship in-wheel, where
    embedder_admin._server_binary() already looks.

    Cleans up the staged copy afterward so the source tree is never left dirty.
    """
    exe = _EMBED_SERVER_BIN + (".exe" if sys.platform.startswith("win") else "")
    cmd = ["cargo", "build", "-p", "m3-embed-server"]
    if release:
        cmd.append("--release")
    if features:
        # The server crate mirrors m3-core-py's feature names (embedded,
        # embedded-cuda/-vulkan/-metal), so the SAME feature list applies.
        cmd += ["--features", ",".join(features)]
    print(f"[build_wheel] building embed server: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_WORKSPACE, env=os.environ.copy())
    if proc.returncode != 0:
        raise SystemExit(
            f"error: `cargo build -p m3-embed-server` failed (exit {proc.returncode}). "
            "The shared-embedder binary is a baseline artifact — the wheel must not "
            "ship without it. Fix the build rather than publishing a binary-less wheel."
        )
    profile = "release" if release else "debug"
    # Honour CARGO_TARGET_DIR. Hardcoding <workspace>/target reads from a
    # directory cargo may not have written to — and then silently ships whatever
    # binary happens to be sitting there. That is not hypothetical: with
    # CARGO_TARGET_DIR redirected (build_local.py does this on Windows to dodge
    # cmake's object-path limit), the vulkan build wrote its 65 MB server to the
    # relocated dir while this line picked up the 138 MB CUDA server left in
    # <workspace>/target by an earlier build. The result was a
    # m3-core-rs-windows-vulkan wheel containing a CUDA embed-server: it passed
    # every existing check (binary present, plausible size, RECORD valid) and
    # was only caught by scanning the binary for backend symbols. Caught before
    # release, 2026-07-26.
    target_root = Path(os.environ.get("CARGO_TARGET_DIR") or (_WORKSPACE / "target"))
    built = target_root / profile / exe
    if not built.is_file():
        raise SystemExit(
            f"error: expected embed-server binary not found at {built}"
            + (f" (CARGO_TARGET_DIR={os.environ['CARGO_TARGET_DIR']})"
               if os.environ.get("CARGO_TARGET_DIR") else "")
        )
    # Assert the binary we are about to ship was actually built for THIS
    # backend. The path fix above closes the known way a mismatched server got
    # staged, but any future one (a silently-skipped rebuild, a stale artifact,
    # a wrong --features) would be just as invisible: the wheel still contains
    # "a binary of about the right size", which is all the downstream checks
    # look at. Scanning for a backend-defining symbol is cheap and turns a
    # silent mis-ship into a build failure.
    _BACKEND_SYMBOL = {
        "embedded-vulkan": (b"ggml_backend_vk", "Vulkan"),
        "embedded-cuda": (b"ggml_backend_cuda", "CUDA"),
        "embedded-metal": (b"ggml_backend_metal", "Metal"),
    }
    for feat, (needle, label) in _BACKEND_SYMBOL.items():
        if feat not in features:
            continue
        blob = built.read_bytes()
        if needle not in blob:
            others = [
                lbl for f, (n, lbl) in _BACKEND_SYMBOL.items()
                if f != feat and n in blob
            ]
            raise SystemExit(
                f"error: {built.name} was built for {features} but contains no "
                f"{label} symbols"
                + (f" (it looks like a {'/'.join(others)} build)" if others else "")
                + f". Refusing to ship a wrong-backend embed-server. Path: {built}"
            )
        break

    _PY_PKG_DIR.mkdir(parents=True, exist_ok=True)
    staged = _PY_PKG_DIR / exe
    shutil.copy2(built, staged)
    print(f"[build_wheel] staged {exe} -> {staged.relative_to(_HERE)}")
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            staged.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True,
                        choices=sorted({b for _, b in _MATRIX}),
                        help="GPU/CPU backend to build")
    parser.add_argument("--os", dest="os_tok", default=None,
                        choices=["windows", "linux", "macos"],
                        help="target OS token (default: inferred from host)")
    parser.add_argument("--out", default=None,
                        help="maturin --out directory (default: dist/<package>)")
    parser.add_argument("--release", action="store_true", default=True,
                        help="build in release profile (default: on)")
    parser.add_argument("maturin_args", nargs="*",
                        help="extra args passed through to `maturin build` "
                             "(put after a literal -- )")
    args = parser.parse_args(argv)

    os_tok = args.os_tok or host_os()
    key = (os_tok, args.backend)
    if key not in _MATRIX:
        valid = ", ".join(f"{o}-{b}" for o, b in sorted(_MATRIX))
        print(f"error: unsupported combination {os_tok}-{args.backend}.\n"
              f"  valid: {valid}", file=sys.stderr)
        return 2

    features = _MATRIX[key]
    name = package_name(os_tok, args.backend)
    out_dir = args.out or str(_HERE.parent.parent / "dist" / name)

    # Invoke maturin via the `maturin` shim when it's on PATH, else fall back to
    # `python -m maturin` (same interpreter). A venv that installed maturin as a
    # dependency but doesn't have the Scripts/ shim on PATH would otherwise fail
    # with a bare-command FileNotFoundError.
    maturin_argv = (
        ["maturin"] if shutil.which("maturin")
        else [sys.executable, "-m", "maturin"]
    )
    cmd = [*maturin_argv, "build", "--out", out_dir]
    if args.release:
        cmd.append("--release")
    if features:
        cmd += ["--features", ",".join(features)]
    cmd += args.maturin_args

    print(f"[build_wheel] package={name} os={os_tok} backend={args.backend} "
          f"features={features or '(none)'}")
    print(f"[build_wheel] {' '.join(cmd)}")

    # Build + stage the shared-embedder binary FIRST (fails loud if it can't
    # build), then build the wheel with the binary present in python-source so
    # maturin bundles it. Both happen under the patched package name.
    with _patched_name(name), _staged_embed_server(features, args.release):
        proc = subprocess.run(cmd, cwd=_HERE, env=os.environ.copy())
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
