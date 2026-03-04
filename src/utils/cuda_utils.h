#pragma once

#include <cuda_runtime.h>
#include <driver_types.h>

#include "libbase/string_utils.h"
#include "utils.h"

#define CUDA_SAFE_CALL(expr) cuda::reportError(expr, __LINE__)
#define CUDA_CHECK_STREAM(expr) cuda::reportError(cudaStreamSynchronize(expr), __LINE__)

namespace cuda
{
  ::std::string formatError(cudaError_t code);

  void reportError(cudaError_t err, int line, const ::std::string& prefix = ::std::string());

  void selectCudaDevice(int argc, char** argv);

  inline size_t compute_grid(size_t n) { return std::min(divCeil(n, (size_t)DEFAULT_GROUP_SIZE), (size_t)MAX_GRID_SIZE); }

  class CudaTimer
  {
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
    cudaStream_t stream_{};

   public:
    explicit CudaTimer(cudaStream_t stream) : stream_(stream)
    {
      CUDA_SAFE_CALL(cudaEventCreate(&start_));
      CUDA_SAFE_CALL(cudaEventCreate(&stop_));
      CUDA_SAFE_CALL(cudaEventRecord(start_, stream_));
    }

    ~CudaTimer()
    {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

    double elapsed()
    {
      CUDA_SAFE_CALL(cudaEventRecord(stop_, stream_));
      CUDA_SAFE_CALL(cudaEventSynchronize(stop_));

      float ms = 0.f;
      CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start_, stop_));
      return ms * 1e-3;
    }
  };
}  // namespace cuda
