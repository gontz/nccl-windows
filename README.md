# NCCL for Windows

**Don't open Issues for general NCCL questions or non Windows related problems. Only Windows specific issues.** Any Issue opened that is not Windows specific will be closed automatically.

### Building from source:

A Visual Studio 2019 or newer is required to launch the compiler x64 environment. The installation path is referred in the instructions as VISUAL_STUDIO_INSTALL_PATH.

CUDA path will be found automatically if you have the bin folder in your PATH, or have the CUDA installation path settled on well-known environment vars like CUDA_ROOT, CUDA_HOME or CUDA_PATH.

If none of these are present, make sure to set the environment variable before starting the build:
set CUDA_HOME=CUDA_INSTALLATION_PATH

1. Open a Command Line (cmd.exe)
2. **Clone the NCCL for Windows repository from nccl-windows branch (NOT MAIN): ```cd C:\ & git clone --single-branch --branch nccl-windows https://github.com/SystemPanic/nccl-windows.git```**
3. Execute (in cmd) ```VISUAL_STUDIO_INSTALL_PATH\VC\Auxiliary\Build\vcvarsall.bat x64```
4. Change the working directory to the cloned repository path, for example: ```cd C:\nccl-windows```
5. Build & install:
```
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

#Replace YOUR_GPUS_ARCHS with your multiple gpu's architectures, for example, RTX 5090 is 120, RTX 3090 is 86. You can mix different archs if you have distinct GPU archs, for example 86;120
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=YOUR_GPUS_ARCHS -DCMAKE_INSTALL_PREFIX=install
cmake --build build --parallel --target install
```

After the build, bin, lib and include will be under C:\nccl-windows\install

---

# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
