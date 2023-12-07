#include <omp.h>

#include <CLI/CLI.hpp>
#include <iostream>

#include "Common/helper_cuda.hpp"

__global__ void warm_up() {}

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

int main(int argc, char** argv) {
  CLI::App app("My C++ App");

  int num_threads = 1;

  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->default_val(1);

  CLI11_PARSE(app, argc, argv);

  omp_set_num_threads(num_threads);

  // ------------------------ CUDA ------------------------
  constexpr auto n = 1 << 20;

  float* in_data;
  float* out_data;

  checkCudaErrors(cudaMallocManaged(&in_data, n * sizeof(float)));
  checkCudaErrors(cudaMallocManaged(&out_data, n * sizeof(float)));

  auto streams = new cudaStream_t[num_threads];
  for (int i = 0; i < num_threads; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  warm_up<<<1, 1>>>();

  cudaEventRecord(start, nullptr);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    printf("Hello from thread %d of %d\n", thread_id, num_threads);

    const auto offset = thread_id * 256;
    kernel1<<<1, 256, 0, streams[thread_id]>>>(
        in_data + offset, out_data + offset, thread_id);
  }

  cudaEventRecord(stop, nullptr);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << " - Elapsed time: " << milliseconds << " ms\n";

  for (int i = 0; i < num_threads; i++) {
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  }
  delete[] streams;

  checkCudaErrors(cudaFree(in_data));
  checkCudaErrors(cudaFree(out_data));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}