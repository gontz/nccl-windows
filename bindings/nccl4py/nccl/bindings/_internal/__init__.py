#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

"""
Internal bindings implementation.

On Linux, this module preloads the NCCL library using cuda-pathfinder if
available, which provides better library discovery across different
environments (conda, system installs, custom CUDA paths, etc.).

If cuda-pathfinder is not available or fails to find NCCL, the Cython
bindings will fall back to direct dlopen("libnccl.so.2") which works
if the library is in standard system paths.

On Windows, cuda-pathfinder does not apply (NVIDIA's NCCL wheels are
Linux-only), so _preload_nccl_windows() searches NCCL_ROOT/NCCL_HOME,
C:\\nccl, and finally PATH for a self-built nccl.dll. If that fails the
Cython loader retries and raises with the full list of paths tried.

See documentation for cuda-pathfinder including the search order at:
https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/generated/cuda.pathfinder.load_nvidia_dynamic_lib.html#cuda.pathfinder.load_nvidia_dynamic_lib
"""

import os
import sys


def _preload_nccl_windows() -> bool:
    """Locate and preload nccl.dll on Windows.

    cuda-pathfinder only knows how to find NCCL shipped in NVIDIA's pip wheels,
    which are Linux-only -- on Windows NCCL is a self-built nccl.dll, so we
    search for it ourselves.

    Doing this here rather than leaving it to the Cython loader buys two
    things: os.add_dll_directory(), so the DLL's own dependencies resolve, and
    ctypes' native wide-char loading, which handles install paths that are not
    representable in the ANSI code page. Once loaded, the Cython loader finds
    the module via GetModuleHandleA("nccl.dll") and never has to search.

    Returns True if nccl.dll is loaded in the process afterwards.
    """
    import ctypes

    candidates = []
    for var in ("NCCL_ROOT", "NCCL_HOME"):
        root = os.environ.get(var)
        if root:
            candidates.append(os.path.join(root, "bin", "nccl.dll"))
            candidates.append(os.path.join(root, "nccl.dll"))
    candidates.append(r"C:\nccl\bin\nccl.dll")

    for path in candidates:
        if not os.path.isfile(path):
            continue
        directory = os.path.dirname(os.path.abspath(path))
        try:
            os.add_dll_directory(directory)
        except (OSError, AttributeError):
            pass
        try:
            ctypes.WinDLL(path)
            return True
        except OSError:
            continue

    # Nothing found by path -- fall back to the standard loader search (PATH).
    try:
        ctypes.WinDLL("nccl.dll")
        return True
    except OSError:
        # Leave it to the Cython loader, which raises with the full
        # list of paths it tried.
        return False


# Optional: Preload NCCL library for better discovery
# This runs before the Cython extensions are loaded, allowing the loader to
# resolve symbols from the already-loaded library.
if sys.platform == "win32":
    _preload_nccl_windows()
else:
    try:
        from cuda.pathfinder import load_nvidia_dynamic_lib

        load_nvidia_dynamic_lib("nccl")
    except ImportError:
        # cuda-python not installed, fall back to Cython's dlopen
        pass
    except Exception:
        # Library not found by pathfinder or other error
        # Fall back to Cython's dlopen - it will provide the error message if needed
        pass
