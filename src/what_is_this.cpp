#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

const auto functor2 = [](float x) {
  float result = 0.0f;
  for (int i = 0; i < 10; i++) {
    result += std::sin(x + i) * std::log10(x + i) / std::sqrt(x + i);
  }
  return result;
};

const auto functor3 = [](float x) { return std::sqrt(x); };

void stage1(float* data, const std::size_t count) {
  std::iota(data, data + count, 0);
}

void stage2(float* data, const std::size_t count) {
  std::transform(data, data + count, data, functor2);
}

void stage3(float* data, const std::size_t count) {
  std::transform(data, data + count, data, functor3);
}

void stage4(float* data, const std::size_t count) {
  constexpr int windowSize = 3;
  std::transform(
      data, data + count, data, [&data, windowSize](const float& element) {
        int index = &element - &data[0];
        int startIndex = std::max(0, index - windowSize + 1);
        int endIndex = index + 1;
        return std::accumulate(data + startIndex, data + endIndex, 0.0f) /
               (endIndex - startIndex);
      });
}

// Function 1: Square Root
void computeSquareRoot(float* data, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
      data[i] = std::sqrt(data[i]);
  }
}

// Function 2: Exponential
void computeExponential(float* data, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
      data[i] = std::exp(data[i]);
  }
}

// Function 3: Trigonometric (Sine)
void computeSine(float* data, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
      data[i] = std::sin(data[i]);
  }
}

// Function 4: Random Numbers
void computeRandomNumbers(float* data, std::size_t size) {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (std::size_t i = 0; i < size; ++i) {
      data[i] = static_cast<float>(std::rand()) / RAND_MAX;
  }
}

int main() {
  constexpr std::size_t count = 10'000'000;
  float* data1 = new float[count];
  std::cout << "Memory consumption: " << count * sizeof(float) / 1024 / 1024
            << " MB\n";

  // prepare two more data

  float* data2 = new float[count];
  float* data3 = new float[count];

  computeRandomNumbers(data2, count);
  computeSquareRoot(data2, count);
  computeExponential(data2, count);
  computeSine(data2, count);

  // stage1(data1, count);
  // // stage1(data2, count);
  // // stage1(data3, count);

  // stage2(data1, count);
  // stage3(data1, count);
  // stage4(data1, count);

  // const std::size_t block_size = 100'000;
  // const std::size_t block_count = count / block_size;

  // for (auto i = 0; i < count; ++i) {
  //   data1[i] = functor3(functor2(functor1(i)));
  // }

  // for (auto i = 0; i < count; ++i) data2[i] = functor1(data2[i]);
  // for (auto i = 0; i < count; ++i) data2[i] = functor2(data2[i]);
  // for (auto i = 0; i < count; ++i) data2[i] = functor3(data2[i]);

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