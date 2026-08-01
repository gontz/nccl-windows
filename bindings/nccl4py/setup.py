import os
import re
import subprocess
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup, Extension


IS_WINDOWS = sys.platform == "win32"

# Platform-specific build knobs.
#   - There is no libdl on Windows; the loader uses LoadLibrary/GetProcAddress
#     out of kernel32, which MSVC links by default.
#   - MSVC spells the standard switch /std:c++14, not -std=c++14.
#   - The internal loader has one implementation per platform; see
#     tools/generate_nccl_windows_pyx.py for how nccl_windows.pyx is produced.
if IS_WINDOWS:
    PLATFORM_LIBRARIES = []
    PLATFORM_COMPILE_ARGS = ["/std:c++14", "/EHsc"]
    LOADER_SUFFIX = "windows"
else:
    PLATFORM_LIBRARIES = ["dl"]
    PLATFORM_COMPILE_ARGS = ["-std=c++14"]
    LOADER_SUFFIX = "linux"

# Check CUDA_HOME is set and is a valid directory
CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    # On Windows the CUDA Toolkit installer sets CUDA_PATH, not CUDA_HOME.
    CUDA_HOME = os.environ.get("CUDA_PATH")
if not CUDA_HOME:
    raise SystemExit("Error: CUDA_HOME is not set")

cuda_path = Path(CUDA_HOME)
if not cuda_path.exists() or not cuda_path.is_dir():
    raise SystemExit(f"Error: CUDA_HOME does not exist or is not a directory: {CUDA_HOME}")
CUDA_INC = str(cuda_path / "include")

ext_modules = [
    "nccl.bindings.nccl"
]

def calculate_modules(module: str):
    module_parts = module.split(".")

    # nccl.bindings.nccl -> nccl/bindings/nccl.pyx
    lowpp_mod = module_parts.copy()
    lowpp_pyx = os.path.join(*lowpp_mod[:-1], f"{lowpp_mod[-1]}.pyx")
    lowpp_mod = ".".join(lowpp_mod)
    lowpp_ext = Extension(
        lowpp_mod,
        sources=[lowpp_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=PLATFORM_COMPILE_ARGS,
        libraries=PLATFORM_LIBRARIES,
    )

    # cy variant: nccl.bindings.nccl -> nccl/bindings/cynccl.pyx
    cy_mod = module_parts.copy()
    cy_mod[-1] = f"cy{cy_mod[-1]}"
    cy_mod_pyx = os.path.join(*cy_mod[:-1], f"{cy_mod[-1]}.pyx")
    cy_mod = ".".join(cy_mod)
    cy_ext = Extension(
        cy_mod,
        sources=[cy_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=PLATFORM_COMPILE_ARGS,
        libraries=PLATFORM_LIBRARIES,
    )

    # internal variant: source is nccl_<platform>.pyx, but published module name is nccl.bindings._internal.nccl
    inter_mod = module_parts.copy()
    inter_mod.insert(-1, "_internal")
    inter_mod_pyx = os.path.join(*inter_mod[:-1], f"{inter_mod[-1]}_{LOADER_SUFFIX}.pyx")
    inter_mod = ".".join(inter_mod)
    inter_ext = Extension(
        inter_mod,
        sources=[inter_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=PLATFORM_COMPILE_ARGS,
        libraries=PLATFORM_LIBRARIES,
    )

    # internal variant: insert _internal and use utils.pyx
    inter_utils_mod = module_parts.copy()
    inter_utils_mod.insert(-1, "_internal")
    inter_utils_mod[-1] = "utils"
    inter_utils_mod_pyx = os.path.join(*inter_utils_mod[:-1], f"{inter_utils_mod[-1]}.pyx")
    inter_utils_mod = ".".join(inter_utils_mod)
    inter_utils_ext = Extension(
        inter_utils_mod,
        sources=[inter_utils_mod_pyx],
        include_dirs=[CUDA_INC],
        language="c++",
        extra_compile_args=PLATFORM_COMPILE_ARGS,
        libraries=PLATFORM_LIBRARIES,
    )

    return lowpp_ext, cy_ext, inter_ext, inter_utils_ext


# Note: the extension attributes are overwritten in build_extension()
ext_modules = [e for ext in ext_modules for e in calculate_modules(ext)]


compiler_directives = {"embedsignature": True, "show_performance_hints": True}


setup(
    ext_modules=cythonize(
        ext_modules,
        verbose=True,
        language_level=3,
        compiler_directives=compiler_directives,
    ),
    zip_safe=False,
    options={"build_ext": {"inplace": False}},
)
