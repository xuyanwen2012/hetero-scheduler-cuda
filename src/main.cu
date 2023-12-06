#include <Common/helper_cuda.hpp>  // helper functions for CUDA error checking and initialization
#include <algorithm>
#include <iostream>
#include <numeric>

__global__ void emptyKernel() {}

__global__ void do_some_work(float* in, float* out, const std::size_t n) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float temp = in[i];
    for (int j = 0; j < 1000; ++j) {
      temp = cos(sin(temp)) * tan(temp);  // Expensive trigonometric operations
    }
    out[i] = temp;
  }
}

__global__ void myKernel() {
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  printf("Thread %d is running on SM %d\n", threadId, blockIdx.x / 4);
}

int main() {
  constexpr std::size_t n = 10'000'000;

  float* data;
  float* out_data;

  checkCudaErrors(cudaMallocManaged(&data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data, n * sizeof(float)));

  // initialize data
  std::iota(data, data + n, 0.0f);

  constexpr std::size_t numThreadsPerBlock = 256;
  constexpr std::size_t numBlocks = 16;  // blocks per SM

  // warmup
  emptyKernel<<<numBlocks, numThreadsPerBlock>>>();

  // -----------------------------------
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, nullptr);

  do_some_work<<<numBlocks, numThreadsPerBlock>>>(data, out_data, n);

  cudaEventRecord(stop, nullptr);

  cudaEventSynchronize(stop);

  // Calculate the elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "Elapsed time: " << milliseconds << " ms\n";

  // -----------------------------------

  // Wait for the kernel to finish
  checkCudaErrors(cudaDeviceSynchronize());

  // peek 10 results
  std::cout << "Results: ";
  std::for_each_n(out_data, 10, [](const auto& x) { std::cout << x << "\n"; });

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  checkCudaErrors(cudaFree(data));
  checkCudaErrors(cudaFree(out_data));

  return 0;
}