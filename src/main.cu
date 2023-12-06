#include <Common/helper_cuda.hpp>  // helper functions for CUDA error checking and initialization
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

#include "omp.h"

__global__ void emptyKernel() {}

// __global__ void do_some_work(float* in, float* out, const std::size_t n) {
//   const auto i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < n) {
//     float temp = in[i];
//     for (int j = 0; j < 1000; ++j) {
//       temp = cos(sin(temp)) * tan(temp);  // Expensive trigonometric
//       operations
//     }
//     out[i] = temp;
//   }
// }

__device__ float do_some_work_func(float temp) {
  for (int j = 0; j < 1000; ++j) {
    temp = cos(sin(temp)) * tan(temp);  // Expensive trigonometric operations
  }
  return temp;
}

__global__ void kernel1(float* in, float* out, int which_sm) {
  const auto i = threadIdx.x;
  out[i] = do_some_work_func(in[i]) + which_sm * 1000000;
}

void measureCudaKernel(std::function<void()> kernelFunc,
                       const char* kernelName = "CUDA Kernel") {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, nullptr);

  kernelFunc();

  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << kernelName << " - Elapsed time: " << milliseconds << " ms\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  constexpr std::size_t n = 10'000'000;

  // Duck has 8 SMs
  // Toucan has 36 SMs
  constexpr auto num_sm = 8;

  float* data;
  float* out_data;
  float* out_data_2;

  checkCudaErrors(cudaMallocManaged(&data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data_2, n * sizeof(float)));

  constexpr auto nthreads = 4;
  constexpr auto numBlocks = 1;
  constexpr auto numThreadsPerBlock = 128;

  cudaStream_t* streams = new cudaStream_t[nthreads];

  // initialize data
  std::iota(data, data + n, 0.0f);

  // constexpr std::size_t numThreadsPerBlock = 256;
  // constexpr std::size_t numBlocks = 16;  // blocks per SM

  // warmup
  emptyKernel<<<numBlocks, numThreadsPerBlock>>>();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, nullptr);

  // measureCudaKernel([&]() {
#pragma omp parallel for
  for (int i = 0; i < nthreads; ++i) {
    printf("Thread %d\n", i);
    const auto offset = i * 128;
    kernel1<<<1, 128, 0, streams[i]>>>(data + offset, out_data + offset, i);
  }

  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << " - Elapsed time: " << milliseconds << " ms\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // });

  // -----------------------------------

  // Wait for the kernel to finish
  // checkCudaErrors(cudaDeviceSynchronize());

  // peek 10 results
  std::cout << "Results: ";
  std::for_each_n(out_data, 10, [](const auto& x) { std::cout << x << "\n"; });

  checkCudaErrors(cudaFree(data));
  checkCudaErrors(cudaFree(out_data));
  checkCudaErrors(cudaFree(out_data_2));

  for (int i = 0; i < nthreads + 1; i++) {
    cudaStreamDestroy(streams[i]);
  }

  delete[] streams;

  return 0;
}