#!/usr/bin/env python3
"""Build m3-core-rs wheels locally on this machine for the supported CPython matrix.

This is the per-machine local build driver for the workflow documented in
`docs/PUBLISHING.md` ("Current workflow — build locally on each platform's box"):
each platform's wheels are built on a real box of that platform, smoke-tested on
hardware, then attached to a GitHub Release. It replaces the older per-OS shell
wrappers (`build_windows_wheels.sh` / `build_linux_wheels.sh`) with one
cross-platform script so the OS-specific logic lives in exactly one place.

It is a thin driver around `build_wheel.py` (the single source of truth for the
(os, backend) -> package-name/features mapping). For each requested backend it
resolves the four interpreters, invokes `build_wheel.py` once, and writes a log.

Usage:
    python build_local.py <backend>...          # one or more of cpu vulkan cuda metal
    python build_local.py all                   # every backend valid for this host OS
    python build_local.py cpu --pythons 3.11 3.12
    python build_local.py vulkan --no-smoke-test

Backends must be valid for the host OS (see build_wheel._MATRIX); invalid combos
are skipped with a warning. Exit code is non-zero if any requested build fails.

Key behaviours that previously lived as tribal knowledge in the shell scripts:

  * Interpreter discovery uses `uv` (a prerequisite on every build box), not a
    hardcoded glob of uv's internal install dir. We prefer `uv python find
    --managed-python` (a clean standalone interpreter) and fall back to a plain
    `uv python find` that REJECTS virtualenv paths — otherwise uv happily hands
    back the active project `.venv` for a matching version, which must never be
    used to build a distributable wheel.

  * On Linux, GPU backends (vulkan/cuda) build NATIVE while cpu uses maturin's
    `--zig` for a portable manylinux tag. Under `--zig`, llama-cpp-sys's cmake
    step cannot find the system Vulkan/CUDA library in zig's isolated sysroot
    (`Could NOT find Vulkan (missing: Vulkan_LIBRARY)`), so GPU backends must
    link the host libs natively. zig is never used on Windows/macOS.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse the single-source-of-truth mapping and helpers from build_wheel.py
# (same directory). Importing avoids re-deriving the matrix in a second place.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_wheel import _MATRIX, host_os, package_name  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_PYTHONS = ("3.11", "3.12", "3.13", "3.14")

# A path is a virtualenv interpreter (never use one to build a wheel) if it sits
# under a venv layout. uv may return the active project venv from a plain
# `uv python find`; we filter those out.
_VENV_MARKERS = (".venv", "virtualenvs", "site-packages")


def _utcnow() -> str:
    # Explicit UTC; avoids the deprecated naive utcnow().
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def crate_version() -> str:
    """Read workspace.package.version from the top-level Cargo.toml.

    maturin derives the wheel version from here (via dynamic=["version"]); we
    read the same value so the output directory name matches the wheel name.
    """
    cargo = _REPO_ROOT / "Cargo.toml"
    in_wp = False
    for line in cargo.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            in_wp = s == "[workspace.package]"
        m = re.match(r'version\s*=\s*"([^"]+)"', s)
        if m and (in_wp or _REPO_ROOT == _HERE):
            return m.group(1)
    # Fallback: first version = "..." anywhere.
    m = re.search(r'^version\s*=\s*"([^"]+)"', cargo.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit("error: could not read version from Cargo.toml")
    return m.group(1)


def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "error: `uv` not found on PATH. uv is required to resolve the build "
            "interpreters. Install it (https://docs.astral.sh/uv/) or run from a "
            "login shell where uv is on PATH."
        )
    return uv


def _looks_like_venv(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return any(m in p for m in _VENV_MARKERS)


def resolve_interpreter(version: str, *, install_missing: bool) -> str | None:
    """Resolve one CPython <version> to an absolute interpreter path via uv.

    Strategy (in order):
      1. `uv python find --managed-python <v>` — a clean uv-managed standalone.
      2. `uv python find <v>` — but reject venv paths (uv may return the active
         project .venv for a matching version, which must not build wheels).
      3. If install_missing: `uv python install <v>` then retry (1).
    Returns the path, or None if nothing acceptable is found.
    """
    uv = _uv()

    def _find(managed: bool) -> str | None:
        cmd = [uv, "python", "find"]
        if managed:
            cmd.append("--managed-python")
        cmd.append(version)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None
        out = r.stdout.strip()
        return out or None

    p = _find(managed=True)
    if p:
        return p

    p = _find(managed=False)
    if p and not _looks_like_venv(p):
        return p

    if install_missing:
        subprocess.run([uv, "python", "install", version], check=False)
        p = _find(managed=True)
        if p:
            return p

    return None


def build_one(
    backend: str,
    os_tok: str,
    version: str,
    interps: list[str],
    *,
    use_zig: bool,
) -> int:
    """Invoke build_wheel.py once for `backend`, tee-ing to a per-backend log."""
    out_dir = _REPO_ROOT / "ci-wheels" / f"local-{version}" / f"{os_tok}-{backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"build-{os_tok}-{backend}.log"

    extra: list[str] = []
    if use_zig:
        extra.append("--zig")
    extra += ["--interpreter", *interps]

    cmd = [
        sys.executable,
        str(_HERE / "build_wheel.py"),
        "--backend",
        backend,
        "--os",
        os_tok,
        "--out",
        str(out_dir),
        "--",
        *extra,
    ]

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"=== {os_tok} {backend} build for m3-core-rs {version} ===\n")
        log.write(f"interpreters: {' '.join(interps)}\n")
        log.write(f"zig: {use_zig}\n")
        log.write(f"out: {out_dir}\n")
        log.write(f"start: {_utcnow()}\n")
        log.flush()
        print(f"[build_local] {os_tok}/{backend}: building ({len(interps)} interpreters, "
              f"zig={use_zig}) -> {out_dir}")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=os.environ.copy())
        log.write(f"end: {_utcnow()}\n")
        log.write(f"maturin exit: {proc.returncode}\n")
        wheels = sorted(out_dir.glob("*.whl"))
        log.write("=== wheels produced ===\n")
        for w in wheels:
            log.write(f"{w.stat().st_size:>12,}  {w.name}\n")

    if proc.returncode == 0:
        print(f"[build_local] {os_tok}/{backend}: OK - {len(wheels)} wheel(s). "
              f"log: {log_path}")
    else:
        print(f"[build_local] {os_tok}/{backend}: FAILED (exit {proc.returncode}). "
              f"see {log_path}", file=sys.stderr)
    return proc.returncode


def smoke_test(os_tok: str, version: str, backend: str) -> bool:
    """Install one built wheel into a throwaway uv venv and check the backend label."""
    out_dir = _REPO_ROOT / "ci-wheels" / f"local-{version}" / f"{os_tok}-{backend}"
    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        print(f"[smoke] {os_tok}/{backend}: no wheel to test", file=sys.stderr)
        return False
    # Prefer a 3.13 wheel; otherwise the first available.
    wheel = next((w for w in wheels if "cp313" in w.name), wheels[0])
    uv = _uv()
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        if subprocess.run([uv, "venv", "--python", "3.13", str(venv)],
                          capture_output=True).returncode != 0:
            subprocess.run([uv, "venv", str(venv)], capture_output=True)
        py = venv / ("Scripts" if os_tok == "windows" else "bin") / (
            "python.exe" if os_tok == "windows" else "python")
        if subprocess.run([uv, "pip", "install", "--python", str(py), str(wheel)],
                          capture_output=True).returncode != 0:
            print(f"[smoke] {os_tok}/{backend}: install failed", file=sys.stderr)
            return False
        r = subprocess.run(
            [str(py), "-c",
             "import m3_core_rs; print(m3_core_rs.embed_backend_label())"],
            capture_output=True, text=True)
        label = r.stdout.strip()
        ok = r.returncode == 0 and label != ""
        print(f"[smoke] {os_tok}/{backend}: import {'OK' if ok else 'FAILED'} "
              f"-> backend={label or '(error)'}")
        return ok


def main(argv: list[str] | None = None) -> int:
    valid_backends = sorted({b for _, b in _MATRIX})
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("backends", nargs="+",
                        help=f"backends to build: {', '.join(valid_backends)}, or 'all'")
    parser.add_argument("--pythons", nargs="+", default=list(_PYTHONS),
                        help=f"CPython versions (default: {' '.join(_PYTHONS)})")
    parser.add_argument("--install-missing", action="store_true",
                        help="`uv python install` any interpreter not already present")
    parser.add_argument("--no-smoke-test", dest="smoke", action="store_false",
                        help="skip the post-build import/backend-label check")
    args = parser.parse_args(argv)

    os_tok = host_os()
    host_backends = [b for (o, b) in _MATRIX if o == os_tok]

    requested = args.backends
    if requested == ["all"]:
        requested = host_backends
    else:
        bad = [b for b in requested if b not in valid_backends]
        if bad:
            parser.error(f"unknown backend(s): {', '.join(bad)}. "
                         f"valid: {', '.join(valid_backends)}")

    # Resolve interpreters once (shared across backends).
    print(f"[build_local] host OS: {os_tok}; resolving interpreters via uv...")
    interps: list[str] = []
    for v in args.pythons:
        p = resolve_interpreter(v, install_missing=args.install_missing)
        if not p:
            print(f"[build_local] MISSING interpreter {v} (no managed/system install "
                  f"found; pass --install-missing to fetch it)", file=sys.stderr)
            return 2
        print(f"[build_local]   {v} -> {p}")
        interps.append(p)

    version = crate_version()
    print(f"[build_local] m3-core-rs version {version}")

    rc_total = 0
    for backend in requested:
        if (os_tok, backend) not in _MATRIX:
            print(f"[build_local] skip {backend}: not valid for {os_tok} "
                  f"(valid here: {', '.join(host_backends)})", file=sys.stderr)
            continue
        # zig only for the Linux CPU build; GPU backends and other OSes go native.
        use_zig = (os_tok == "linux" and backend == "cpu")
        rc = build_one(backend, os_tok, version, interps, use_zig=use_zig)
        rc_total = rc_total or rc
        if rc == 0 and args.smoke:
            smoke_test(os_tok, version, backend)

    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
