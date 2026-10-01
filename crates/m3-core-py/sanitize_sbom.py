#!/usr/bin/env python3
"""Strip the builder's local paths out of a wheel's maturin-generated SBOM.

maturin writes each workspace crate's ABSOLUTE source path into the `bom-ref`
of `<dist-info>/sboms/m3-core-py.cyclonedx.json`. On a CI runner that is a
harmless `/home/runner/work/...`; on a developer box it is the builder's home
directory, and it ships in a PUBLIC release artifact:

    path+file:///Users/<user>/m3-core-rs/crates/m3-error#3.10.1
    path+file:///C:/Users/<user>/.m3-dev/m3-core-rs/crates/m3-core-py#3.10.1

57 occurrences per wheel across 226 components — username and full local
directory layout. Found 2026-09-20 in 10 hand-built wheels of v2026.9.20.

A `bom-ref` only has to be internally consistent WITHIN the document (it is a
document-local anchor, not a resolvable URL), so rewriting every occurrence of
the repo-root prefix to CI's canonical form preserves the SBOM's meaning and
leaves every other field untouched.

⚠ RECORD is updated in the same pass, and this coupling is the whole point:
RECORD carries a sha256 + size for every file in the wheel, so editing the SBOM
without recomputing its RECORD row makes `pip install` fail an integrity check.
Editing one without the other produces a wheel that verifies structurally and
then dies at install time. Keep them together in any rewrite of this file.

Rewriting the zip also preserves each entry's stored Unix mode, or it would
undo the m3-embed-server exec bit that build_wheel.py sets (see _set_exec_bit
there, and check 5 in verify_wheels.py).

Idempotent: a wheel with no local paths is left byte-identical and reported as
already clean.

    python sanitize_sbom.py dist/                  # sanitize in place
    python sanitize_sbom.py --check dist/          # report only, exit 1 if dirty
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# What CI produces, and therefore what a sanitized wheel should look like.
_CI_PREFIX = "path+file:///home/runner/work/m3-core-rs/m3-core-rs"

# path+file:///<anything>/crates/<crate>#<version>  ->  capture the prefix so it
# can be swapped wholesale. Non-greedy up to the LAST /crates/ so a builder whose
# checkout itself sits under a directory named "crates" still rewrites correctly.
_LOCAL_PATH_RE = re.compile(r"path\+file:///(?P<root>.*?)(?P<tail>/crates/[^#\"]*)")


def _record_row(data: bytes) -> str:
    """RECORD's hash column: urlsafe base64 sha256, '=' padding stripped."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + digest.decode("ascii").rstrip("=")


def _sanitize_sbom_bytes(raw: bytes) -> tuple[bytes, int]:
    """Return (rewritten, occurrences). Byte-identical when already clean."""
    text = raw.decode("utf-8")
    hits = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal hits
        # Already-generic CI paths are left alone so the pass is idempotent and
        # a mixed dist/ (some CI wheels, some local) is safe to run over.
        if m.group("root").startswith("home/runner/"):
            return m.group(0)
        hits += 1
        return _CI_PREFIX + m.group("tail")

    out = _LOCAL_PATH_RE.sub(_sub, text)
    if not hits:
        return raw, 0
    # Re-serialize through json to prove the rewrite produced a valid document
    # rather than silently corrupting the SBOM; dump the ORIGINAL formatting by
    # returning the substituted text, not json.dumps, so the diff stays minimal.
    json.loads(out)
    return out.encode("utf-8"), hits


def sanitize_wheel(path: Path, *, check_only: bool = False) -> int:
    """Rewrite one wheel in place. Returns the number of paths scrubbed."""
    with zipfile.ZipFile(path) as z:
        sbom_name = next(
            (n for n in z.namelist() if n.endswith("cyclonedx.json")), None
        )
        if sbom_name is None:
            return 0
        new_sbom, hits = _sanitize_sbom_bytes(z.read(sbom_name))

    if hits == 0 or check_only:
        return hits

    record_name = next(
        n for n in zipfile.ZipFile(path).namelist() if n.endswith(".dist-info/RECORD")
    )
    new_row = f"{sbom_name},{_record_row(new_sbom)},{len(new_sbom)}"

    # Rewrite into a temp file, then atomically replace, so an interrupted run
    # cannot leave a half-written wheel that later steps would happily upload.
    tmp = Path(tempfile.mkstemp(suffix=".whl", dir=str(path.parent))[1])
    try:
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED
        ) as dst:
            for info in src.infolist():
                if info.filename == sbom_name:
                    data = new_sbom
                elif info.filename == record_name:
                    rows = src.read(info).decode("utf-8").splitlines()
                    data = (
                        "\n".join(
                            new_row if r.startswith(sbom_name + ",") else r
                            for r in rows
                        )
                        + "\n"
                    ).encode("utf-8")
                else:
                    data = src.read(info)
                # Carry external_attr across verbatim: it stores the Unix mode,
                # so dropping it would strip the m3-embed-server exec bit.
                out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                out.external_attr = info.external_attr
                out.internal_attr = info.internal_attr
                out.create_system = info.create_system
                out.compress_type = info.compress_type
                dst.writestr(out, data)
        shutil.copystat(path, tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return hits


def _wheels(targets: list[str]) -> list[Path]:
    found: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.whl")))
        elif p.suffix == ".whl":
            found.append(p)
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="+", help="wheel files and/or directories")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report leaked paths without rewriting; exit 1 if any are found",
    )
    args = ap.parse_args(argv)

    wheels = _wheels(args.targets)
    if not wheels:
        print(f"no wheels found under: {', '.join(args.targets)}", file=sys.stderr)
        return 2

    dirty = 0
    for w in wheels:
        hits = sanitize_wheel(w, check_only=args.check)
        if hits:
            dirty += 1
            verb = "LEAKS" if args.check else "scrubbed"
            print(f"[{verb}] {w.name}  ({hits} local path refs)")
        else:
            print(f"[ clean ] {w.name}")

    print("-" * 60)
    if args.check:
        print(f"{dirty} of {len(wheels)} wheel(s) still carry local paths")
        return 1 if dirty else 0
    print(f"sanitized {dirty} of {len(wheels)} wheel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
