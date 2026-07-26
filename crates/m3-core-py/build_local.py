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
import tempfile
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


def _vcvars_env() -> dict[str, str] | None:
    """Environment produced by running MSVC's ``vcvars64.bat``, or None.

    Windows GPU backends need the MSVC toolchain on PATH (cl.exe, link.exe, the
    Windows SDK) before llama-cpp-sys's cmake step runs. A plain shell has none
    of it, so the build dies deep inside cmake with errors that do not name the
    real cause. Rather than require every caller to remember to launch from a
    "x64 Native Tools" prompt, run vcvars64 in a subshell and lift the resulting
    environment — the same thing that prompt does.

    Discovery order: VSINSTALLDIR (already inside a dev shell), then vswhere
    (the supported API, finds any edition), then the known BuildTools/Community
    paths. Returns None when MSVC is not installed; the caller then proceeds
    unchanged so a CPU-only box is unaffected.
    """
    if os.name != "nt":
        return None

    candidates: list[Path] = []
    vsdir = os.environ.get("VSINSTALLDIR")
    if vsdir:
        candidates.append(Path(vsdir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) \
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        try:
            found = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
            if found:
                candidates.append(Path(found) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        except Exception:  # noqa: BLE001 — vswhere is best-effort discovery
            pass

    pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        candidates.append(pf86 / "Microsoft Visual Studio" / "2022" / edition
                          / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    bat = next((c for c in candidates if c.is_file()), None)
    if bat is None:
        return None

    # `set` after vcvars64 dumps the fully-populated environment. The marker
    # separates vcvars' own banner from the variable list.
    #
    # Driven through a temp .bat rather than `cmd /c "call ... && set"`: the
    # inline form is brittle here — quoting/redirection inside the one-liner
    # made cmd exit 1 with "The system cannot find the path specified" before
    # vcvars even ran, while calling the same script on its own succeeded.
    # A script file sidesteps cmd's parsing entirely.
    #
    # cwd is pinned to %SystemRoot%: this process may be launched from a path
    # cmd cannot resolve (e.g. a Git-bash-style path), which makes cmd fail on
    # startup regardless of what it was asked to do.
    marker = "__M3_VCVARS_DONE__"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", delete=False,
                                         encoding="ascii") as fh:
            fh.write("@echo off\n")
            fh.write(f'call "{bat}" >nul 2>&1\n')
            fh.write(f"echo {marker}\n")
            fh.write("set\n")
            tmp = fh.name
        proc = subprocess.run(
            ["cmd", "/c", tmp],
            capture_output=True, text=True, timeout=300,
            cwd=os.environ.get("SystemRoot", r"C:\Windows"),
        )
    except Exception:  # noqa: BLE001 — never let discovery break the build
        return None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    if proc.returncode != 0 or marker not in proc.stdout:
        return None

    env: dict[str, str] = {}
    for line in proc.stdout.split(marker, 1)[1].splitlines():
        key, sep, val = line.partition("=")
        if sep and key:
            env[key] = val
    return env or None


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

    env = os.environ.copy()
    # Strip build-host paths from debuginfo / panic strings so published wheels
    # don't leak the builder's $HOME, cargo registry path, or repo location. The
    # three remappings cover: the cargo registry (crate sources rustc embeds in
    # panic-location strings), $CARGO_HOME, and the repo root itself.
    home = os.path.expanduser("~")
    cargo_home = os.environ.get("CARGO_HOME", os.path.join(home, ".cargo"))
    rust_remaps = [
        f"--remap-path-prefix={cargo_home}=/cargo",
        f"--remap-path-prefix={str(_REPO_ROOT)}=/m3-core-rs",
        f"--remap-path-prefix={home}=/home",
    ]
    existing_rustflags = env.get("RUSTFLAGS", "").strip()
    env["RUSTFLAGS"] = (existing_rustflags + " " + " ".join(rust_remaps)).strip()
    # llama-cpp-sys compiles its C/C++ sources via cmake; rustc's
    # --remap-path-prefix doesn't reach those. Pass the equivalent
    # -ffile-prefix-map to clang/gcc through CFLAGS / CXXFLAGS so __FILE__
    # macros and DWARF debug info are scrubbed there too.
    cc_remaps = " ".join([
        f"-ffile-prefix-map={cargo_home}=/cargo",
        f"-ffile-prefix-map={str(_REPO_ROOT)}=/m3-core-rs",
        f"-ffile-prefix-map={home}=/home",
    ])
    for var in ("CFLAGS", "CXXFLAGS"):
        env[var] = (env.get(var, "").strip() + " " + cc_remaps).strip()
    extra: list[str] = []
    if use_zig:
        extra.append("--zig")

    # Windows GPU: the MSVC toolchain + the Ninja generator, both required.
    #
    # 1. MSVC must be on PATH before llama-cpp-sys's cmake step runs, which a
    #    plain shell does not provide.
    # 2. The Visual Studio CMake generator hits a CMake-4.x + MSBuild
    #    ExternalProject batch-label bug (`VCEnd`) while building
    #    vulkan-shaders-gen — the build fails with "The system cannot find the
    #    batch label specified - VCEnd" and MSB8066. Forcing Ninja avoids it.
    # 3. glslc (Vulkan SDK) must be on PATH for shader compilation.
    #
    # This was documented in WHEEL_BUILD_GUIDE.md §3b as something the operator
    # had to set up by hand, and CI did it separately — so a local build failed
    # for anyone who had not read that far. Doing it here means the driver is
    # self-sufficient on Windows, matching what it already does for Linux CUDA
    # below. Explicit env still wins: nothing is overwritten if the caller
    # already ran vcvars (setdefault + only prepending to PATH).
    if os_tok == "windows" and backend in ("cuda", "vulkan"):
        vc = _vcvars_env()
        if vc:
            for key, val in vc.items():
                if key.upper() == "PATH":
                    continue
                env.setdefault(key, val)
            # PATH must MERGE, not be replaced: the caller's PATH carries uv,
            # cargo and python, which vcvars64 knows nothing about.
            vc_path = next((v for k, v in vc.items() if k.upper() == "PATH"), "")
            if vc_path:
                env["PATH"] = vc_path + os.pathsep + env.get("PATH", "")
        else:
            print(f"[build_local] WARNING: MSVC (vcvars64) not found — {backend} "
                  "will likely fail in llama-cpp-sys's cmake step. Install VS "
                  "Build Tools with the C++ workload.")

        env.setdefault("CMAKE_GENERATOR", "Ninja")

        # glslc ships in the Vulkan SDK's Bin dir and is not on PATH by default.
        sdk = env.get("VULKAN_SDK")
        if sdk:
            sdk_bin = os.path.join(sdk, "Bin")
            if os.path.isdir(sdk_bin) and sdk_bin.lower() not in env.get("PATH", "").lower():
                env["PATH"] = sdk_bin + os.pathsep + env.get("PATH", "")
        elif backend == "vulkan":
            print("[build_local] WARNING: VULKAN_SDK is unset — glslc will not be "
                  "found and the Vulkan shader build will fail.")

    # Linux CUDA needs several things the generic path doesn't (see PUBLISHING.md
    # "Linux CUDA build recipe"):
    #   * nvcc's .cu.o objects are non-PIC but link into a cdylib, so force PIC;
    #   * `libcuda.so.1` is the driver stub that must NOT be bundled, so skip the
    #     auditwheel repair that would try to vendor it;
    #   * the CUDA runtime is statically linked, so the wheel is huge (~475 MB)
    #     unless we (a) trim the GPU-arch fatbin and (b) strip symbols. We cap the
    #     arch list to the GPUs we support; llama-cpp-sys forwards any CMAKE_* env
    #     var to the llama.cpp cmake build, so CMAKE_CUDA_ARCHITECTURES reaches it.
    #     CARGO_PROFILE_RELEASE_STRIP makes rustc strip the final cdylib.
    if os_tok == "linux" and backend == "cuda":
        # nvcc compiles llama.cpp's ggml-cuda/*.cu kernels and embeds their
        # absolute source paths in DWARF — and -ffile-prefix-map in CFLAGS/
        # CXXFLAGS does NOT reach nvcc, so without this the CUDA .so leaks
        # $HOME (the builder's username) ~190x via paths like
        # /home/<user>/.cargo/registry/.../ggml-cuda/argsort.cu. Forward the
        # same remaps to nvcc's host compiler via -Xcompiler so the .cu paths
        # are scrubbed too. -fPIC is required (nvcc .cu.o links into a cdylib).
        cuda_xcompiler = " ".join(
            f"-Xcompiler -ffile-prefix-map={src}={dst}"
            for src, dst in ((cargo_home, "/cargo"), (str(_REPO_ROOT), "/m3-core-rs"), (home, "/home"))
        )
        env.setdefault("CMAKE_CUDA_FLAGS", f"-Xcompiler -fPIC {cuda_xcompiler}")
        env.setdefault("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        # We ship a STATIC CUDA wheel: cuBLAS/cudart are linked into the .so so
        # the wheel is self-contained and Just Works for users without a system
        # CUDA runtime install. It is large (~470-580 MB) — too big for PyPI's
        # 100 MiB limit, so Linux CUDA is distributed via GitHub Release assets
        # (no size cap), not PyPI. (A dynamic-link build to shrink it was tried
        # 2026-06-08 and rejected: it only got to 464 MB because LLAMA_BUILD_
        # SHARED_LIBS doesn't flip cuBLAS off static, and it would burden users
        # with a CUDA-runtime prerequisite. Static is the better tradeoff.)
        # strip symbols is a free ~20-40% win with no downside.
        env.setdefault("CARGO_PROFILE_RELEASE_STRIP", "symbols")
        # libcuda.so.1 is the driver stub and must NOT be bundled; we vendor no
        # CUDA libs, so skip the auditwheel repair entirely.
        extra += ["--auditwheel", "skip"]

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
        if os_tok == "linux" and backend == "cuda":
            log.write(f"CMAKE_CUDA_FLAGS: {env.get('CMAKE_CUDA_FLAGS')}\n")
            log.write("link: STATIC cuBLAS/cudart (self-contained; GitHub Release, not PyPI)\n")
            log.write(f"strip: {env.get('CARGO_PROFILE_RELEASE_STRIP')}\n")
            log.write("auditwheel: skip\n")
        log.write(f"out: {out_dir}\n")
        log.write(f"start: {_utcnow()}\n")
        log.flush()
        print(f"[build_local] {os_tok}/{backend}: building ({len(interps)} interpreters, "
              f"zig={use_zig}) -> {out_dir}")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"end: {_utcnow()}\n")
        log.write(f"maturin exit: {proc.returncode}\n")
        wheels = sorted(out_dir.glob("*.whl"))
        log.write("=== wheels produced ===\n")
        for w in wheels:
            log.write(f"{w.stat().st_size:>12,}  {w.name}\n")

    if proc.returncode == 0:
        sizes = [w.stat().st_size for w in wheels]
        avg_mb = (sum(sizes) / len(sizes) / 1048576) if sizes else 0
        print(f"[build_local] {os_tok}/{backend}: OK - {len(wheels)} wheel(s), "
              f"~{avg_mb:.0f} MB each. log: {log_path}")
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
