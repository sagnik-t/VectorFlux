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

# ── Include paths ──────────────────────────────────────────────────────────────
include_directories(include)

# ── Extension module ───────────────────────────────────────────────────────────
# Each task adds its source files here.
# T01: hello-world CUDA kernel + pybind11 wiring
# T02: Tensor CPU class
# T04: Basic CPU ops (add, mul, relu, matmul)
pybind11_add_module(_core
    python/bindings.cpp
    src/hello.cu
    src/tensor.cpp      # T02: Tensor CPU class
    src/ops_cpu.cpp     # T04: Basic CPU ops
)

set_target_properties(_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Drop the .so directly into the Python package folder
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/vectorflux
)
