#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// A trivial kernel — each thread writes its ID into an array
__global__ void hello_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx * 2;
    }
}

// CPU-side launcher — called from Python
std::string hello_cuda() {
    const int n = 8;
    int* d_out;

    cudaMalloc(&d_out, n * sizeof(int));

    hello_kernel<<<1, n>>>(d_out, n);
    cudaDeviceSynchronize();

    int h_out[n];
    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::string result = "CUDA kernel output: [";
    for (int i = 0; i < n; i++) {
        result += std::to_string(h_out[i]);
        if (i < n - 1) result += ", ";
    }
    result += "]";
    return result;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "VectorFlux core extension";
    m.def("hello_cuda", &hello_cuda, "Run a hello-world CUDA kernel");
}