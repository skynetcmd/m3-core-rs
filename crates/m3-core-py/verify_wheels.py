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
import re
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
        # All 35 shipped 3.9.20 wheels store this binary non-executable (0600 on
        # linux/macos, 0644 on windows). v2026.9.16 and v2026.9.7 are clean
        # 0755, and the correction below does not fire on any current local
        # toolchain (maturin 1.13.3 macos / 1.15.0 linux both PRESERVE the
        # mode) — so this is not "maturin always drops the bit": it is specific
        # to how v2026.9.20 was built, the one release built through CI.
        #
        # ⚠ An earlier note here claimed pip 25.1.1 installs the binary
        # non-executable (0o664) and that m3's runtime chmod therefore stays
        # load-bearing. That was measured against a 0600 wheel and blamed the
        # installer for the artifact's defect. Re-measured 2026-10-01 against a
        # correct 0755 wheel, every pip tested preserves it:
        #     pip 25.1.1 / 25.2 / 26.0 / 26.2.1 -> -rwxr-xr-x
        #     uv                                 -> -rwxr-xr-x
        # pip reproduces whatever the wheel stores. Ship 0755 and the installed
        # binary is executable on every installer tested, which is what makes a
        # downstream workaround unnecessary rather than merely redundant.
        # (m3's `_ensure_executable` / `repair_exec_bit` still earn their keep
        # for users who already installed a 3.9.20 wheel.)
        #
        # A silent regression to 0600 would otherwise stay invisible until a
        # user's embed-server failed to start with EACCES — a symptom that
        # names no cause. selftest() proves this check actually fires.
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
                    "Every installer tested reproduces the stored mode, so "
                    "this ships as EACCES at first use; set it "
                    "on the wheel entry after the build (build_wheel.py)."
                )

        # 6. The SBOM must not carry the builder's absolute paths.
        #
        # maturin writes each crate's absolute source path into its bom-ref —
        # 57 refs per wheel exposing the builder's username and directory
        # layout in a PUBLIC artifact. v2026.9.20 shipped 10 such wheels
        # because the scrub was a manual step someone had to remember.
        # build_wheel.py now does it automatically; this is the assertion that
        # makes forgetting it impossible rather than merely unlikely.
        sbom = next((n for n in z.namelist() if n.endswith("cyclonedx.json")), None)
        if sbom is not None:
            text = z.read(sbom).decode("utf-8", "replace")
            leaked = {
                m.group(0)
                for m in re.finditer(r"path\+file:///[^\"#]*", text)
                if not m.group(0).startswith("path+file:///home/runner/")
            }
            if leaked:
                sample = sorted(leaked)[0]
                problems.append(
                    f"SBOM leaks {len(leaked)} builder path(s), e.g. {sample} "
                    "— run sanitize_sbom.py before publishing"
                )
    return problems



def _synthetic_wheel(
    dirpath: Path, *, mode: int, win: bool = False, sbom_root: str | None = None
) -> Path:
    """A minimal but VALID cpu wheel, so the only variable is the one under test.

    Padded past the 2 MB cpu floor (`_MIN_BIN_MB`) and given a correct RECORD,
    because verify_wheel checks those first — a wheel that trips an earlier
    check would make this prove nothing about the mode check.

    `sbom_root` adds a CycloneDX SBOM whose bom-ref is rooted at that path, to
    exercise check 6 (pass CI's `/home/runner/...` for the clean case).
    """
    import base64 as _b64
    import hashlib as _hashlib
    import zipfile as _zip

    plat = "win_amd64" if win else "manylinux_2_38_x86_64"
    exe = "m3-embed-server.exe" if win else "m3-embed-server"
    name = f"m3_core_rs_{'windows' if win else 'linux'}_cpu-9.9.9-cp313-cp313-{plat}.whl"
    whl = dirpath / name
    binary = b"\0" * (3 * 1024 * 1024)          # clear the 2 MB cpu floor
    so = b"\0" * 1024
    members = {
        f"m3_core_rs/{exe}": binary,
        "m3_core_rs/m3_core_rs.cpython-313-x86_64-linux-gnu.so": so,
        "m3_core_rs/__init__.py": b"__version__ = '9.9.9'\n",
    }
    dist = "m3_core_rs_linux_cpu-9.9.9.dist-info"
    if sbom_root is not None:
        import json as _json

        members[f"{dist}/sboms/m3-core-py.cyclonedx.json"] = _json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "components": [
                    {
                        "type": "library",
                        "name": "m3-error",
                        "bom-ref": f"path+file://{sbom_root}/crates/m3-error#9.9.9",
                    }
                ],
            }
        ).encode()
    rows = []
    for n, data in members.items():
        d = _b64.urlsafe_b64encode(_hashlib.sha256(data).digest()).decode().rstrip("=")
        rows.append(f"{n},sha256={d},{len(data)}")
    rows.append(f"{dist}/RECORD,,")
    with _zip.ZipFile(whl, "w", _zip.ZIP_DEFLATED) as z:
        for n, data in members.items():
            info = _zip.ZipInfo(n)
            info.external_attr = (mode << 16)
            z.writestr(info, data)
        z.writestr(f"{dist}/WHEEL", "Wheel-Version: 1.0\n")
        z.writestr(f"{dist}/METADATA", "Metadata-Version: 2.1\nName: m3-core-rs-linux-cpu\nVersion: 9.9.9\n")
        z.writestr(f"{dist}/RECORD", "\n".join(rows) + "\n")
    return whl


def selftest() -> int:
    """Prove the exec-bit check fires, and only when it should.

    A check that cannot fail is worth nothing, and this one guards a defect that
    is invisible until a user's embed-server dies with EACCES. Run with
    `--selftest`; no network, no build, works on every supported OS.
    """
    import tempfile

    failures = []

    def expect(label: str, cond: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        (d / "bad").mkdir(parents=True)
        bad = _synthetic_wheel(d / "bad", mode=0o600)
        probs = verify_wheel(bad)
        mode_probs = [p for p in probs if "not executable" in p or "NO unix mode" in p]
        expect("a 0o600 binary is reported as not executable", bool(mode_probs), f"problems={probs}")
        expect("and nothing ELSE is wrong with the synthetic wheel",
               probs == mode_probs, f"unexpected: {[p for p in probs if p not in mode_probs]}")

        (d / "good").mkdir(parents=True)
        good = _synthetic_wheel(d / "good", mode=0o755)
        probs = verify_wheel(good)
        expect("a 0o755 binary passes cleanly", probs == [], f"problems={probs}")

        (d / "win").mkdir(parents=True)
        w = _synthetic_wheel(d / "win", mode=0o600, win=True)
        probs = verify_wheel(w)
        expect("Windows wheels are exempt (no exec bit there)",
               not [p for p in probs if "not executable" in p], f"problems={probs}")

        # Check 6: the SBOM leak guard must fire on a builder path and stay
        # quiet on CI's generic one, or it is decoration.
        (d / "leak").mkdir(parents=True)
        leak = _synthetic_wheel(
            d / "leak", mode=0o755, sbom_root="/Users/somebody/m3-core-rs"
        )
        probs = verify_wheel(leak)
        expect("an SBOM with a builder path is reported as leaking",
               any("SBOM leaks" in p for p in probs), f"problems={probs}")

        (d / "noleak").mkdir(parents=True)
        clean = _synthetic_wheel(
            d / "noleak",
            mode=0o755,
            sbom_root="/home/runner/work/m3-core-rs/m3-core-rs",
        )
        probs = verify_wheel(clean)
        expect("a CI-rooted SBOM passes cleanly", probs == [], f"problems={probs}")

        # And the sanitizer must actually turn the first into the second.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import sanitize_sbom

        dirty = _synthetic_wheel(
            d / "leak", mode=0o755, sbom_root="/Users/somebody/m3-core-rs"
        )
        scrubbed = sanitize_sbom.sanitize_wheel(dirty)
        probs = verify_wheel(dirty)
        expect("sanitize_sbom clears the leak and keeps the wheel valid",
               scrubbed > 0 and probs == [], f"scrubbed={scrubbed} problems={probs}")

    print()
    if failures:
        print(f"selftest FAILED: {len(failures)} check(s) — {failures}")
        return 1
    print("selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--selftest":
        return selftest()
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
