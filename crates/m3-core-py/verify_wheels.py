#!/usr/bin/env python3
"""Verify m3-core-rs wheels ship the m3-embed-server shared-embedder binary.

Every published wheel must bundle BOTH native artifacts (see docs/BUILD_WHEELS.md):
  1. the Python extension  m3_core_rs/m3_core_rs.*.{pyd,so}
  2. the shared server bin m3_core_rs/m3-embed-server[.exe]

This checks EVERY wheel under the given dir(s), asserts both are present, that
the RECORD lists the binary with a correct sha256 + size (so pip install won't
reject it), and that the binary size is backend-appropriate (a cuda wheel must
not ship a tiny CPU-only server — the exact backend-mismatch footgun). Exit 1 if
any wheel fails. Pure stdlib; runs on any build box (Windows/Linux/macOS).

Usage:
    python verify_wheels.py <dir> [<dir>...]        # scan wheel dirs
    python verify_wheels.py ci-wheels/local-3.6.27  # e.g. this box's output
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
import zipfile
from pathlib import Path

# Minimum plausible server-binary size per backend token found in the wheel
# name. cuda/vulkan STATICALLY link a GPU llama.cpp and are far larger than the
# ~8 MB CPU server, so a value below the floor means the wrong backend was
# bundled. Floors are deliberately loose (well under observed sizes: cpu ~8 MB,
# vulkan ~68 MB, cuda ~145 MB) to catch a mismatch, not to pin exact sizes.
#
# METAL IS DIFFERENT: on macOS the GPU backend is Apple's Metal.framework,
# linked DYNAMICALLY (a system framework), so nothing GPU is baked into the
# binary — a correct metal server is ~8 MB, same order as CPU. So a size floor
# CANNOT distinguish a real metal build from a CPU one for macOS; metal gets a
# CPU-like floor here and the real proof is framework linkage (`otool -L` shows
# Metal/MetalKit/Accelerate — verified by hand on the 3.7.4 8.2 MB binary). A
# too-high metal floor false-failed correct wheels, which is why it is 2, not 10.
_MIN_BIN_MB = {"cpu": 2, "vulkan": 30, "cuda": 60, "metal": 2}


def _backend_of(wheel_name: str) -> str | None:
    for tok in ("cpu", "cuda", "vulkan", "metal"):
        if f"_{tok}-" in wheel_name or f"-{tok}-" in wheel_name:
            return tok
    return None


def _record_entry(z: zipfile.ZipFile, target: str) -> tuple[str, int] | None:
    """Return (sha256_b64, size) that RECORD claims for target, or None."""
    rec = next((n for n in z.namelist() if n.endswith(".dist-info/RECORD")), None)
    if not rec:
        return None
    text = z.read(rec).decode("utf-8")
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        path = row[0]
        if path == target:
            digest = row[1] if len(row) > 1 else ""
            size = int(row[2]) if len(row) > 2 and row[2] else 0
            return digest, size
    return None


def verify_wheel(path: Path) -> list[str]:
    """Return a list of problem strings (empty = wheel OK)."""
    problems: list[str] = []
    exe = "m3-embed-server.exe" if "win" in path.name else "m3-embed-server"
    target = f"m3_core_rs/{exe}"
    with zipfile.ZipFile(path) as z:
        names = z.namelist()

        # 1. Python extension present.
        if not any(n.endswith((".pyd", ".so")) and "m3_core_rs" in n for n in names):
            problems.append("no Python extension (.pyd/.so)")

        # 2. Server binary present.
        if target not in names:
            found = [n for n in names if "m3-embed-server" in n]
            problems.append(
                f"missing {target}" + (f" (found instead: {found})" if found else "")
            )
            return problems  # nothing more to check without the binary

        info = z.getinfo(target)
        raw = z.read(target)

        # 3. Backend-appropriate size.
        backend = _backend_of(path.name)
        mb = info.file_size / (1024 * 1024)
        floor = _MIN_BIN_MB.get(backend or "", 0)
        if mb < floor:
            problems.append(
                f"{backend} server only {mb:.1f} MB (< {floor} MB floor) "
                "— wrong backend bundled?"
            )

        # 4. RECORD lists it with a correct sha256 + size (else pip rejects it).
        claimed = _record_entry(z, target)
        if claimed is None:
            problems.append("binary not listed in RECORD (pip install would ignore/err)")
        else:
            digest_b64, size = claimed
            actual = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(raw).digest()
            ).decode().rstrip("=")
            if size != info.file_size:
                problems.append(f"RECORD size {size} != actual {info.file_size}")
            if digest_b64 and digest_b64 != actual:
                problems.append("RECORD sha256 mismatch (corrupt/edited wheel)")

        # 5. The server binary is stored EXECUTABLE in the wheel.
        #
        # Measured 2026-10-01 on the shipped 3.9.20 wheels: maturin stores
        # staged python-source files at mode 0o600, discarding the 0755 cargo
        # produced. `shutil.copy2` in build_wheel.py is not the culprit — it
        # preserves mode; the loss is in wheel packaging.
        #
        # ⚠ Read this before concluding the check makes the problem go away.
        # The INSTALLER decides whether the stored mode survives:
        #     pip 25.1.1 -> 0o664, NOT executable   (pip only marks
        #                                            *.data/scripts/ entries +x)
        #     uv         -> 0o775, executable
        # m3 installs via pipx, which uses pip. So shipping 0755 fixes the uv
        # path and leaves the pip path needing m3's runtime chmod
        # (`_ensure_executable` before exec, `repair_exec_bit` under
        # `m3 doctor --fix`) — those stay load-bearing.
        #
        # This check exists anyway, for two reasons worth the four lines: a
        # wheel that ships 0755 is correct for every installer that honours it,
        # and a silent regression back to 0600 would otherwise be invisible
        # until a user's embed-server failed to start with EACCES — four
        # downstream symptoms that name no cause.
        #
        # Windows has no exec bit (`os.chmod` ignores 0o111 and `os.access`
        # X_OK is true for every file), so asserting one there would be
        # vacuously green at best and a false failure at worst.
        if "win" not in path.name:
            mode = (info.external_attr >> 16) & 0o7777
            if mode == 0:
                problems.append(
                    f"{target} has NO unix mode recorded in the wheel "
                    "(external_attr empty) — it cannot be executable on install"
                )
            elif not mode & 0o111:
                problems.append(
                    f"{target} is stored mode {oct(mode)} — not executable. "
                    "maturin drops cargo's 0755; set it on the wheel entry after "
                    "the build. uv honours the stored mode, pip does not, so m3's "
                    "runtime chmod is still required for pip/pipx installs."
                )
    return problems


def main(argv: list[str]) -> int:
    dirs = [Path(a) for a in argv] or [Path("ci-wheels")]
    wheels: list[Path] = []
    for d in dirs:
        wheels += sorted(d.rglob("*.whl")) if d.is_dir() else ([d] if d.suffix == ".whl" else [])
    if not wheels:
        print(f"no wheels found under: {', '.join(str(d) for d in dirs)}", file=sys.stderr)
        return 2

    ok = bad = 0
    for w in wheels:
        problems = verify_wheel(w)
        exe = "m3-embed-server.exe" if "win" in w.name else "m3-embed-server"
        try:
            with zipfile.ZipFile(w) as z:
                sz = z.getinfo(f"m3_core_rs/{exe}").file_size / (1024 * 1024)
                size_s = f"{sz:6.1f} MB"
        except KeyError:
            size_s = "   --   "
        if problems:
            bad += 1
            print(f"[FAIL] {w.name}")
            for p in problems:
                print(f"         - {p}")
        else:
            ok += 1
            print(f"[ OK ] {w.name}  (server {size_s})")

    print("-" * 60)
    print(f"{ok} OK, {bad} FAILED, {len(wheels)} total")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
