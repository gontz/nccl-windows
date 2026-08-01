# nccl4py on Windows

Upstream nccl4py is Linux-only. This directory carries a Windows port on top of
it. This file documents what was changed and how to build/install it.

## Install

Prerequisites: the NCCL DLL from this repo's build (`cmake --build build --target
install`), CUDA 12.x or 13.x, Python 3.10+, and an MSVC toolchain (the same one
used to build `nccl.dll`).

```powershell
# 1. Put the built install tree somewhere permanent, outside the git checkout
robocopy C:\nccl-windows\install C:\nccl /E

# 2. Point at it (User scope; reopen your shell afterwards)
[Environment]::SetEnvironmentVariable('NCCL_ROOT','C:\nccl','User')
[Environment]::SetEnvironmentVariable('NCCL_HOME','C:\nccl','User')
$p=[Environment]::GetEnvironmentVariable('Path','User')
[Environment]::SetEnvironmentVariable('Path',"$p;C:\nccl\bin",'User')

# 3. Build and install the bindings
$env:CUDA_HOME = $env:CUDA_PATH
cd C:\nccl-windows\bindings\nccl4py
python -m pip install "cuda-bindings~=13.0"   # or ~=12.0 for CUDA 12.x
python -m pip install .
```

Do **not** use the `[cu12]`/`[cu13]` extras on Windows. They pull
`nvidia-nccl-cu12`/`nvidia-nccl-cu13`, which publish manylinux wheels only and
would in any case supply a Linux NCCL you cannot use. Install `cuda-bindings`
directly instead, as above; the rest of the dependency set
(`cuda.core`, `cuda-pathfinder`, `numpy`, `packaging`) resolves normally.

Verify:

```python
from nccl.bindings import nccl as b
v = b.get_version()
print(f"{v//10000}.{(v//100)%100}.{v%100}")
```

## How nccl.dll is located

In order:

1. already loaded in the process (`GetModuleHandleA`)
2. `%NCCL_ROOT%\bin\nccl.dll`, then `%NCCL_ROOT%\nccl.dll` — same for `%NCCL_HOME%`
3. `C:\nccl\bin\nccl.dll`
4. bare `nccl.dll`, i.e. the standard loader search (PATH, app dir, …)

Steps 2–4 run twice: once in Python (`_internal/__init__.py`) and once in the
Cython loader. The Python pass runs first and does the real work — it calls
`os.add_dll_directory()` so the DLL's own dependencies resolve, and it loads via
`ctypes.WinDLL`, which handles paths that aren't representable in the ANSI code
page. The Cython loader then finds the module at step 1. If the Python pass
found nothing, the Cython pass retries and raises with the full list of paths it
tried.

`nccl.dll` links only the MSVC runtime plus `WS2_32`/`IPHLPAPI`; the CUDA runtime
is statically linked and the driver is loaded on demand, so no `cudart64_*.dll`
needs to sit beside it.

## What was changed

### `nccl/bindings/_internal/nccl_windows.pyx` (generated)

Upstream ships only `nccl_linux.pyx` — `dlopen`/`dlsym` against `libnccl.so.2`,
with `get_cuda_version()` going through `libcuda.so.1`. That file is
machine-generated and is regenerated whenever the NCCL API changes, so the port
is written as a **transform**, not a hand-maintained fork:

```powershell
python tools\generate_nccl_windows_pyx.py
```

Re-run that after any upstream refresh instead of re-porting by hand. It rewrites
two things and copies the rest through verbatim:

- the *Extern* section — `<dlfcn.h>` is replaced by a small C shim over
  `LoadLibraryA`/`GetProcAddress`/`GetModuleHandleA`/`GetLastError`, and
  `get_cuda_version()` reads `cuDriverGetVersion` from `nvcuda.dll`.
- the symbol table in `_check_or_init_nccl()` — Windows has no `RTLD_DEFAULT`
  (no process-wide symbol namespace), so upstream's "try the global namespace,
  then fall back to the handle" pair collapses to a single `GetProcAddress`
  against an explicit module handle.

`_inspect_function_pointers()` and all 39 `_nccl*` wrapper thunks are portable
and pass through untouched. The generator asserts on its own output: it fails if
the section banners move, if it doesn't replace exactly one `load_library()`, if
no symbol blocks match, or if any `dl*`/`RTLD_` token survives.

Two details worth knowing if you touch it:

- The Win32 calls go through a C shim rather than being declared straight to
  Cython, because `GetProcAddress` returns `FARPROC` (a function pointer) and
  Cython emits C++ here, where the implicit `FARPROC` → `void*` conversion is
  illegal. The cast has to happen in C.
- The shim contains no path building and no backslashes. Path search lives in
  Cython/Python where `os.path` and real error messages are available.

### `nccl/bindings/_internal/__init__.py`

Added `_preload_nccl_windows()` and gated the cuda-pathfinder preload behind
`sys.platform != "win32"` — pathfinder only knows how to find NCCL inside
NVIDIA's pip wheels, which don't exist for Windows.

### `setup.py`

- `libraries=["dl"]` → `[]` on Windows. There is no libdl; the loader uses
  kernel32, which MSVC links by default.
- `-std=c++14` → `/std:c++14 /EHsc` on Windows (MSVC spelling).
- The internal loader source is selected per platform:
  `nccl_{linux,windows}.pyx`.
- `CUDA_HOME` falls back to `CUDA_PATH`, which is what the CUDA Toolkit
  installer actually sets on Windows.

`nccl/core/` — the entire high-level API — needed no changes; it is pure Python
with no platform assumptions.

## Verified on

NCCL 2.29.7, CUDA 13.3, Python 3.13, RTX 5090 (sm_120) + RTX 3090 (sm_86).

- Single process, `ncclCommInitAll` across both GPUs: AllReduce over 1M floats.
- Two processes (`spawn`), high-level `nccl.core` API, unique_id passed over a
  multiprocessing queue: AllReduce, Reduce-to-root, Broadcast, Send/Recv.

Note that the bundled `examples/` use `mpi4py` and expect `mpirun`. That's a
convention of the examples, not a requirement of the library — any mechanism
that gets the unique_id from rank 0 to the other ranks works, including
`torch.distributed`, a file, or a socket.

## Not covered

This ports the bindings only. It does **not** give stock PyTorch NCCL support on
Windows: the official Windows wheels are built with `USE_NCCL=0`, so
`torch.distributed.is_nccl_available()` stays `False` no matter where `nccl.dll`
sits. That would require rebuilding PyTorch from source with `USE_NCCL=1` and
`USE_SYSTEM_NCCL=1` against `NCCL_ROOT`.
