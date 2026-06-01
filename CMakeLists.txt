cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

project(vectorflux LANGUAGES CXX CUDA)

# ── Standards ──────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Auto-detect your GPU's compute capability (RTX 5060 → sm_120)
set(CMAKE_CUDA_ARCHITECTURES native)

# ── Python ─────────────────────────────────────────────────────────────────────
# Picks up the active .venv automatically
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ── pybind11 ───────────────────────────────────────────────────────────────────
add_subdirectory(third_party/pybind11)

# ── CUDA toolkit (gives .cpp files access to cuda_runtime.h) ──────────────────
# LANGUAGES CUDA only routes .cu files through nvcc; plain .cpp files compiled
# by g++ need explicit include/link via the CUDAToolkit imported target.
find_package(CUDAToolkit REQUIRED)

# ── Include paths ──────────────────────────────────────────────────────────────
include_directories(include)

# ── Extension module ───────────────────────────────────────────────────────────
# T01: hello-world CUDA kernel + pybind11 wiring
# T02: Tensor CPU class
# T04: Basic CPU ops (add, mul, relu, matmul)
# T05: CUDA tensor backend (.to())
# T06: Element-wise CUDA kernels (add, mul, relu)
# T07: cuBLAS matmul
# T08: Op base class + DAG graph nodes
# T09: Forward pass / Session.run()
pybind11_add_module(_core
    python/bindings.cpp
    src/hello.cu
    src/tensor.cpp      # T02/T05: Tensor class (CPU + CUDA backend)
    src/ops_cpu.cpp     # T04: CPU ops + dispatch layer
    src/ops_cuda.cu     # T06/T07: CUDA kernels + cuBLAS matmul
    src/graph.cpp       # T08: Op base class + DAG graph nodes
    src/session.cpp     # T09: Forward pass / Session.run()
)

# CUDA::cudart  — cuda_runtime.h + libcudart (needed by tensor.cpp via g++)
# CUDA::cublas  — cublas_v2.h   + libcublas (needed by ops_cuda.cu)
target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Drop the .so directly into the Python package folder
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/vectorflux
)
