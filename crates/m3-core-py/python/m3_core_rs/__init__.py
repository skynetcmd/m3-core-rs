"""m3_core_rs — Rust core for m3-memory (Project Oxidation).

This package re-exports symbols from the compiled `.pyd` (Rust extension).

Windows-specific note: when built with a GPU feature (`embedded-cuda`,
`embedded-vulkan`, `embedded-metal`), the compiled `.pyd` dynamically links
against vendor runtime DLLs (e.g. `cublas64_*.dll`, `cublasLt64_*.dll`,
`cudart64_*.dll` for CUDA). Python on Windows since 3.8 does NOT use `PATH`
for DLL resolution from extension modules — only directories registered via
`os.add_dll_directory()` are searched. We register the standard CUDA
install location automatically when `CUDA_PATH` is set in the environment.

For other GPU backends or non-standard install locations, operators can set
`M3_EMBED_DLL_DIRS` (semicolon-separated on Windows, colon-separated on
POSIX) to additional directories to pre-register. These are added before
the import of the compiled extension.

Failing to find the DLLs at import time raises `ImportError: DLL load
failed while importing m3_core_rs`. See `docs/EMBED_DEPLOYMENT.md`.
"""

import os as _os
import sys as _sys


def _register_dll_dirs() -> None:
    # Only meaningful on Windows; on POSIX, ld.so reads LD_LIBRARY_PATH.
    if _sys.platform != "win32":
        return
    if not hasattr(_os, "add_dll_directory"):
        return  # pre-3.8, nothing to do

    candidates: list[str] = []

    # CUDA: prefer $CUDA_PATH if set in the process environment; otherwise
    # walk the standard NVIDIA install root and pick the newest installed
    # CUDA version. Children of GUI-launched processes (IDEs, services,
    # scheduled tasks) often don't inherit Machine-level env vars, so the
    # fallback is load-bearing for them.
    cuda_paths: list[str] = []
    cuda_path = _os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        cuda_paths.append(cuda_path)
    nvidia_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if _os.path.isdir(nvidia_root):
        try:
            versions = [d for d in _os.listdir(nvidia_root)
                        if d.startswith("v") and _os.path.isdir(_os.path.join(nvidia_root, d))]
            # Lexicographic sort works for CUDA's vMAJOR.MINOR scheme up to v9.x;
            # this is fine until CUDA v100 ships. Add a numeric fallback later.
            versions.sort(reverse=True)
            for v in versions:
                cuda_paths.append(_os.path.join(nvidia_root, v))
        except OSError:
            pass

    for cp in cuda_paths:
        candidates.append(_os.path.join(cp, "bin", "x64"))  # CUDA 13.x layout
        candidates.append(_os.path.join(cp, "bin"))         # CUDA 12.x and older

    # Operator override: M3_EMBED_DLL_DIRS=<dir>;<dir>;... (Windows ; separator).
    extra = _os.environ.get("M3_EMBED_DLL_DIRS", "").strip()
    if extra:
        for d in extra.split(_os.pathsep):
            d = d.strip()
            if d:
                candidates.append(d)

    seen: set[str] = set()
    for d in candidates:
        if d in seen or not _os.path.isdir(d):
            continue
        seen.add(d)
        try:
            _os.add_dll_directory(d)
        except (OSError, ValueError):
            # Best-effort — if we can't register a dir we'll surface the
            # ImportError from the extension load below.
            pass


_register_dll_dirs()

from .m3_core_rs import *  # noqa: E402,F401,F403

# Mirror the maturin-default __doc__ / __all__ propagation so users see
# the Rust crate's docstring on `help(m3_core_rs)`.
from . import m3_core_rs as _ext  # noqa: E402

__doc__ = _ext.__doc__
if hasattr(_ext, "__all__"):
    __all__ = _ext.__all__
