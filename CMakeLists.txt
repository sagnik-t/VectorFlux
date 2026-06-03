cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-13)

project(vectorflux LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(third_party/pybind11)

find_package(CUDAToolkit REQUIRED)

include_directories(include)

# T01: hello-world CUDA kernel + pybind11 wiring
# T02: Tensor CPU class
# T04: Basic CPU ops (add, mul, relu, matmul)
# T05: CUDA tensor backend (.to())
# T06: Element-wise CUDA kernels (add, mul, relu)
# T07: cuBLAS matmul
# T08: Op base class + DAG graph nodes
# T09: Forward pass / Session.run()
# T10: Autograd backward pass (step, transpose, gradients)
pybind11_add_module(_core
    python/bindings.cpp
    src/hello.cu
    src/tensor.cpp
    src/ops_cpu.cpp
    src/ops_cuda.cu
    src/graph.cpp
    src/session.cpp
    src/autograd.cpp
)

target_link_libraries(_core PRIVATE CUDA::cudart CUDA::cublas)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/vectorflux
)
