#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Generates nccl/bindings/_internal/nccl_windows.pyx from the upstream,
# auto-generated nccl_linux.pyx.
#
# Upstream ships only a Linux loader (dlopen/dlsym against libnccl.so.2). The
# file is machine-generated and regenerated whenever the NCCL API changes, so
# this port is written as a transform rather than a hand-maintained fork:
# re-run it after every upstream refresh instead of re-porting by hand.
#
# Only two things are platform-specific:
#   1. the "Extern" section  -- <dlfcn.h> + get_cuda_version() via libcuda.so.1
#   2. load_library() and the dlsym() symbol-resolution pairs in
#      _check_or_init_nccl()
# _inspect_function_pointers() and every _nccl* wrapper thunk are portable and
# are copied through verbatim.
#
# Usage:  python tools/generate_nccl_windows_pyx.py

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERNAL = HERE.parent / "nccl" / "bindings" / "_internal"
SRC = INTERNAL / "nccl_linux.pyx"
DST = INTERNAL / "nccl_windows.pyx"

# --------------------------------------------------------------------------
# The Windows replacement for the "Extern" section.
#
# The Win32 calls are wrapped in a small C shim rather than declared directly
# to Cython. Two reasons:
#   - GetProcAddress returns FARPROC (a function pointer). Cython emits C++
#     here, and C++ forbids the implicit FARPROC -> void* conversion, so the
#     cast has to happen in C.
#   - windows.h drags in min/max/etc. macros; WIN32_LEAN_AND_MEAN + NOMINMAX
#     keep it contained.
#
# The shim deliberately contains no path building and no backslashes -- the
# DLL search is done in Cython where os.path and real error messages are
# available.
# --------------------------------------------------------------------------
WINDOWS_EXTERN = '''\
###############################################################################
# Extern
###############################################################################

# You must 'from .utils import NotSupportedError' before using this template

cdef extern from *:
    """
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
    #include <windows.h>

    /* Cast through an integer so C++ never sees an implicit FARPROC -> void*. */
    static void* nccl4py_get_symbol(void* handle, const char* name) {
        return (void*)(INT_PTR)GetProcAddress((HMODULE)handle, name);
    }

    static void* nccl4py_load_named(const char* name) {
        return (void*)LoadLibraryA(name);
    }

    static void* nccl4py_get_module(const char* name) {
        return (void*)GetModuleHandleA(name);
    }

    static unsigned long nccl4py_last_error(void) {
        return (unsigned long)GetLastError();
    }
    """
    void* nccl4py_get_symbol(void* handle, const char* name) nogil
    void* nccl4py_load_named(const char* name) nogil
    void* nccl4py_get_module(const char* name) nogil
    unsigned long nccl4py_last_error() nogil


cdef inline void* get_symbol(void* handle, const char* name) noexcept nogil:
    return nccl4py_get_symbol(handle, name)


cdef int get_cuda_version():
    cdef void* handle = NULL
    cdef void* cuDriverGetVersion = NULL
    cdef int err, driver_ver = 0

    # Load the CUDA driver to check its version
    handle = nccl4py_load_named(b"nvcuda.dll")
    if handle == NULL:
        raise NotSupportedError(
            f'CUDA driver is not found (LoadLibrary("nvcuda.dll") failed, '
            f'GetLastError={nccl4py_last_error()})'
        )
    cuDriverGetVersion = nccl4py_get_symbol(handle, b"cuDriverGetVersion")
    if cuDriverGetVersion == NULL:
        raise RuntimeError('Did not find cuDriverGetVersion symbol in nvcuda.dll')
    err = (<int (*)(int*) noexcept nogil>cuDriverGetVersion)(&driver_ver)
    if err != 0:
        raise RuntimeError(f'cuDriverGetVersion returned error code {err}')

    return driver_ver

'''

# --------------------------------------------------------------------------
# Windows replacement for load_library().
#
# Search order (matches the documented contract in README-windows.md):
#   1. nccl.dll already loaded in this process (e.g. preloaded by
#      _internal/__init__.py, which also handles non-ANSI paths)
#   2. %NCCL_ROOT%\\bin, %NCCL_HOME%\\bin  (then the prefix root itself)
#   3. C:\\nccl\\bin
#   4. bare "nccl.dll" -> the standard loader search order (PATH, app dir, ...)
#
# There is no RTLD_DEFAULT equivalent on Windows, so unlike the Linux loader
# every symbol is resolved from an explicit module handle.
# --------------------------------------------------------------------------
WINDOWS_LOAD_LIBRARY = '''\
cdef void* _load_nccl_handle() except NULL:
    """Locate and load nccl.dll. Requires the GIL."""
    cdef void* handle = NULL
    cdef bytes encoded

    # Already loaded in this process (preloaded by _internal/__init__.py, or
    # pulled in by another extension) -- reuse it rather than searching.
    handle = nccl4py_get_module(b"nccl.dll")
    if handle != NULL:
        return handle

    candidates = []
    for var in ("NCCL_ROOT", "NCCL_HOME"):
        root = os.environ.get(var)
        if root:
            candidates.append(os.path.join(root, "bin", "nccl.dll"))
            candidates.append(os.path.join(root, "nccl.dll"))
    candidates.append(os.path.join("C:", os.sep, "nccl", "bin", "nccl.dll"))
    candidates.append("nccl.dll")

    for cand in candidates:
        # LoadLibraryA is ANSI; non-representable paths are handled by the
        # ctypes preload in _internal/__init__.py, which lands us in the
        # GetModuleHandleA fast path above.
        try:
            encoded = cand.encode("mbcs")
        except UnicodeEncodeError:
            continue
        handle = nccl4py_load_named(encoded)
        if handle != NULL:
            return handle

    raise RuntimeError(
        "Failed to load nccl.dll. Tried:\\n  "
        + "\\n  ".join(candidates)
        + f"\\n(last GetLastError={nccl4py_last_error()})\\n"
        "Set NCCL_ROOT to your NCCL install prefix -- the directory that "
        "contains bin\\\\nccl.dll -- or put that bin directory on PATH."
    )


cdef void* load_library() except* nogil:
    cdef void* handle = NULL
    with gil:
        handle = _load_nccl_handle()
    return handle
'''

# Upstream's Linux load_library(), matched so we can swap it out wholesale.
LINUX_LOAD_LIBRARY_RE = re.compile(
    r"^cdef void\* load_library\(\) except\* nogil:\n"
    r"(?:[ \t]+.*\n|\n)*?"
    r"[ \t]+return handle\n",
    re.M,
)

# The per-symbol resolution pair emitted by upstream's generator:
#
#     global __ncclFoo
#     __ncclFoo = dlsym(RTLD_DEFAULT, 'ncclFoo')
#     if __ncclFoo == NULL:
#         if handle == NULL:
#             handle = load_library()
#         __ncclFoo = dlsym(handle, 'ncclFoo')
#
# Windows has no RTLD_DEFAULT (no process-wide symbol namespace), so the
# global-lookup attempt collapses and we always resolve from the handle.
SYMBOL_RE = re.compile(
    r"^(?P<indent>[ \t]*)global (?P<g>__\w+)\n"
    r"[ \t]*(?P=g) = dlsym\(RTLD_DEFAULT, '(?P<sym>\w+)'\)\n"
    r"[ \t]*if (?P=g) == NULL:\n"
    r"[ \t]*if handle == NULL:\n"
    r"[ \t]*handle = load_library\(\)\n"
    r"[ \t]*(?P=g) = dlsym\(handle, '(?P=sym)'\)\n",
    re.M,
)


def _replace_symbol(m: "re.Match[str]") -> str:
    i, g, sym = m.group("indent"), m.group("g"), m.group("sym")
    return (
        f"{i}global {g}\n"
        f"{i}if handle == NULL:\n"
        f"{i}    handle = load_library()\n"
        f"{i}{g} = get_symbol(handle, '{sym}')\n"
    )


def main() -> int:
    if not SRC.is_file():
        sys.exit(f"error: source not found: {SRC}")

    text = SRC.read_text(encoding="utf-8")

    # Split off the module header (imports) and keep everything from the
    # "Wrapper init" banner onward -- that tail holds the symbol table,
    # _inspect_function_pointers(), and all the _nccl* thunks.
    extern_banner = "###############################################################################\n# Extern\n"
    init_banner = "###############################################################################\n# Wrapper init\n"
    if extern_banner not in text or init_banner not in text:
        sys.exit("error: could not find section banners; upstream layout changed")

    header = text.split(extern_banner, 1)[0]
    tail = init_banner + text.split(init_banner, 1)[1]

    # os is needed by _load_nccl_handle(); upstream only imports threading.
    if "\nimport os\n" not in header:
        header = header.replace("\nimport threading\n", "\nimport os\nimport threading\n", 1)

    header = header.replace(
        "# This code was automatically generated",
        "# Windows port generated by tools/generate_nccl_windows_pyx.py from\n"
        "# nccl_linux.pyx. Do not edit directly -- edit the generator and re-run it.\n"
        "#\n"
        "# The Linux original was automatically generated",
        1,
    )

    # Passing a function, not a string: re.sub() would otherwise expand escape
    # sequences in the replacement template, turning the literal \n inside the
    # generated error message into real newlines and breaking the .pyx.
    tail, n_load = LINUX_LOAD_LIBRARY_RE.subn(lambda _m: WINDOWS_LOAD_LIBRARY, tail)
    if n_load != 1:
        sys.exit(f"error: expected exactly 1 load_library() definition, replaced {n_load}")

    tail, n_syms = SYMBOL_RE.subn(_replace_symbol, tail)
    if n_syms == 0:
        sys.exit("error: no dlsym symbol blocks matched; upstream layout changed")

    out = header + WINDOWS_EXTERN + "\n" + tail

    leftovers = [t for t in ("dlopen", "dlsym", "dlerror", "dlclose", "RTLD_", "dlfcn") if t in out]
    if leftovers:
        sys.exit(f"error: Linux-only tokens survived the transform: {leftovers}")

    DST.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {DST}")
    print(f"  {n_syms} symbols rewritten to GetProcAddress, load_library() replaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
