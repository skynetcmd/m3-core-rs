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
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PYPROJECT = _HERE / "pyproject.toml"

# The canonical build matrix. Key: (os, backend). Value: maturin Cargo
# features to activate (empty for the plain CPU build — the default crate
# build links no llama.cpp GPU backend). The package name is derived as
# `m3-core-rs-<os>-<backend>` for every entry; keeping the name implicit
# guarantees it matches what the wizard computes from the same two tokens.
#
# macOS is Metal-only: Apple Silicon always has a Metal GPU, so a CPU-only
# mac package would be pointless and is intentionally absent.
_MATRIX: dict[tuple[str, str], list[str]] = {
    ("windows", "cpu"): [],
    ("windows", "cuda"): ["embedded-cuda"],
    ("windows", "vulkan"): ["embedded-vulkan"],
    ("linux", "cpu"): [],
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

    cmd = ["maturin", "build", "--out", out_dir]
    if args.release:
        cmd.append("--release")
    if features:
        cmd += ["--features", ",".join(features)]
    cmd += args.maturin_args

    print(f"[build_wheel] package={name} os={os_tok} backend={args.backend} "
          f"features={features or '(none)'}")
    print(f"[build_wheel] {' '.join(cmd)}")

    with _patched_name(name):
        proc = subprocess.run(cmd, cwd=_HERE, env=os.environ.copy())
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
