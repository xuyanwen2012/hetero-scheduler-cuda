#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>


const auto functor1 = [](float x) { return x * x; };
const auto functor2 = [](float x) { return x / 2; };
const auto functor3 = [](float x) { return x + 100; };


void stage1(float* data, const std::size_t count) {
  std::iota(data, data + count, 0);
}

void stage2(float* data, const std::size_t count) {
  std::transform(data, data + count, data, functor1);
}

void stage3(float* data, const std::size_t count) {
  std::transform(data, data + count, data, functor2);
}

void stage4(float* data, const std::size_t count) {
  std::transform(data, data + count, data, functor3);
}

int main() {
  const std::size_t count = 10'000'000;
  float* data1 = new float[count];
  std::cout << "Memory consumption: " << count * sizeof(float) / 1024 / 1024
            << " MB\n";
  
  // prepare two more data 

  float* data2 = new float[count];
  float* data3 = new float[count];  

  stage1(data1, count);
  stage1(data2, count);
  stage1(data3, count);

  // const std::size_t block_size = 100'000;
  // const std::size_t block_count = count / block_size;

  for (auto i = 0; i < count; ++i) {
    data1[i] = functor3(functor2(functor1(i)));
  }

  for (auto i = 0; i < count; ++i) data2[i] = functor1(data2[i]);
  for (auto i = 0; i < count; ++i) data2[i] = functor2(data2[i]);
  for (auto i = 0; i < count; ++i) data2[i] = functor3(data2[i]);
  


  // peek 10 results
  std::cout << "Results:\n";
  for (int i = 0; i < 10; ++i) {
    std::cout << data1[i] << "\n";
  }

  std::cout << "Results:\n";
  for (int i = 0; i < 10; ++i) {
    std::cout << data2[i] << "\n";
  }

  std::cout << "Results:\n";
  for (int i = 0; i < 10; ++i) {
    std::cout << data3[i] << "\n";
  }


  delete[] data1;
  delete[] data2;
  delete[] data3;

  return 0;
}