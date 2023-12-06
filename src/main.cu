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

  int numSMs;
  cudaGetDeviceCount(&numSMs);

  // print how many SMs on this GPU
  std::cout << "Number of SMs: " << numSMs << std::endl;

  if (numSMs < 1) {
    std::cerr << "No SMs found on the GPU." << std::endl;
    return 1;
  }

  int targetSM = 0;  // Targeting the first SM (SM 0)

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,
                          0);  // Assuming you are using the first GPU

  int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
  int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;

  std::cout << "Max threads per SM: " << maxThreadsPerSM << std::endl;
  std::cout << "Max blocks per SM: " << maxBlocksPerSM << std::endl;

  int numThreadsPerBlock = 256;    // Choose an appropriate value
  int numBlocks = maxBlocksPerSM;  // Use all blocks on the target SM

  if (numThreadsPerBlock > maxThreadsPerSM) {
    std::cerr
        << "The selected number of threads per block exceeds the SM's limit."
        << std::endl;
    return 1;
  }

  if (numBlocks > maxBlocksPerSM) {
    std::cerr << "The selected number of blocks exceeds the SM's limit."
              << std::endl;
    return 1;
  }

  // print how many memory used
  std::cout << "Memory used: " << n * sizeof(float) / 1024 / 1024 << " MB"
            << std::endl;

  // initialize data
  std::iota(data, data + n, 0.0f);

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