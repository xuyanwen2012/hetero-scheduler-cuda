#include <omp.h>

#include <CLI/CLI.hpp>
#include <Common/helper_cuda.hpp>  // helper functions for CUDA error checking and initialization
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

__global__ void EmptyKernel() {}

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

__global__ void kernel1(const float* in, float* out, const int which_sm) {
  const auto i = threadIdx.x;
  out[i] = do_some_work_func(in[i]) + which_sm * 1000000;
}

void MeasureCudaKernel(const std::function<void()>& kernel_func,
                       const char* kernel_name = "CUDA Kernel") {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, nullptr);

  kernel_func();

  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << kernel_name << " - Elapsed time: " << milliseconds << " ms\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
  CLI::App app("My C++ App");

  int num_threads = 1;

  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->default_val(1);

  CLI11_PARSE(app, argc, argv);


  constexpr std::size_t n = 10'000'000;

  // Duck has 8 SMs
  // Toucan has 36 SMs
  // constexpr auto num_sm = 8;

  float* data;
  float* out_data;
  float* out_data_2;

  checkCudaErrors(cudaMallocManaged(&data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data_2, n * sizeof(float)));

  auto streams = new cudaStream_t[num_threads];
  for (int i = 0; i < num_threads; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  // initialize data
  std::iota(data, data + n, 0.0f);

  // constexpr std::size_t numThreadsPerBlock = 256;
  // constexpr std::size_t numBlocks = 16;  // blocks per SM

  // warmup
  EmptyKernel<<<1, 1>>>();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, nullptr);

  omp_set_num_threads(num_threads);

  #pragma omp parallel
  // for (int i = 0; i < num_threads; ++i) {
  {
    unsigned int cpu_thread_id = omp_get_thread_num();
    unsigned int num_cpu_threads = omp_get_num_threads();

    printf("Thread %d is running\n", cpu_thread_id);

    const auto offset = cpu_thread_id * 128;
    kernel1<<<1, 128, 0, streams[cpu_thread_id]>>>(
        data + offset, out_data + offset, cpu_thread_id);
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

  for (int i = 0; i < num_threads; i++) {
    cudaStreamDestroy(streams[i]);
  }

  delete[] streams;

  return 0;
}