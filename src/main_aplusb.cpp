#include <cuda_runtime_api.h>
#include <libbase/stats.h>
#include <libbase/timer.h>

#include <fstream>

#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include "utils/utils.h"

void run(int argc, char** argv)
{
  cuda::selectCudaDevice(argc, argv);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  size_t n = 100u * 1000u * 1000u;

  std::vector<unsigned int> as(n);
  std::vector<unsigned int> bs(n);

  for (size_t i = 0; i < n; ++i) {
    as[i] = 3 * (i + 5) + 7;
    bs[i] = 11 * (i + 13) + 17;
  }

  unsigned int *a_gpu, *b_gpu, *c_gpu;
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&a_gpu), n * sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&b_gpu), n * sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&c_gpu), n * sizeof(unsigned int)));

  CUDA_SAFE_CALL(cudaMemcpy(a_gpu, as.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(b_gpu, bs.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));

  std::vector<double> times;
  for (int iter = 0; iter < 10; ++iter) {
    cudaEvent_t start, stop;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&stop));

    CUDA_SAFE_CALL(cudaEventRecord(start));

    cuda::aplusb(stream, a_gpu, b_gpu, c_gpu, n);

    CUDA_SAFE_CALL(cudaEventRecord(stop));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));

    times.push_back(ms / 1000.0);

    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(stop));
  }

  double med = stats::median(times);
  double memory_size_gb = sizeof(unsigned int) * 3.0 * n / 1024.0 / 1024.0 / 1024.0;

  std::cout << "a + b kernel median time (s): " << med << std::endl;
  std::cout << "a + b kernel median VRAM bandwidth: " << memory_size_gb / med << " GB/s" << std::endl;

  std::vector<unsigned int> cs(n);
  CUDA_SAFE_CALL(cudaMemcpy(cs.data(), c_gpu, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < n; ++i) {
    if (cs[i] != as[i] + bs[i]) {
      throw std::runtime_error("Verification failed at index " + std::to_string(i));
    }
  }

  std::cout << "Verification passed!" << std::endl;

  CUDA_SAFE_CALL(cudaFree(a_gpu));
  CUDA_SAFE_CALL(cudaFree(b_gpu));
  CUDA_SAFE_CALL(cudaFree(c_gpu));
}

int main(int argc, char** argv)
{
  try {
    run(argc, argv);
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
