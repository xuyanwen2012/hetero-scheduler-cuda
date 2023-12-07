#include <omp.h>

#include <CLI/CLI.hpp>
#include <iostream>

int main(int argc, char** argv) {
  CLI::App app("My C++ App");

  int num_threads = 1;

  app.add_option("-t,--threads", num_threads, "Number of threads")
      ->default_val(1);

  CLI11_PARSE(app, argc, argv);

  omp_set_num_threads(num_threads);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    printf("Hello from thread %d of %d\n", thread_id, num_threads);
  }

  return 0;
}