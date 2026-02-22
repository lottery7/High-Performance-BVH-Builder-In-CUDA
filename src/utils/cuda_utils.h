#pragma once

#include <driver_types.h>

#include "libbase/string_utils.h"

namespace cuda
{
  std::string formatError(cudaError_t code);

  void reportError(cudaError_t err, int line, const std::string& prefix = std::string());

  void checkKernelErrors(cudaStream_t stream, int line);
  void checkKernelErrors(cudaStream_t stream, int line, bool synchronized);

  void selectCudaDevice(int argc, char** argv);

#define CUDA_SAFE_CALL(expr) cuda::reportError(expr, __LINE__)
#define CUDA_TRACE(expr) expr;
#define CUDA_CHECK_KERNEL(stream) cuda::checkKernelErrors(stream, __LINE__)
#define CUDA_CHECK_KERNEL_SYNC(stream) cuda::checkKernelErrors(stream, __LINE__, true)
#define CUDA_CHECK_KERNEL_ASYNC(stream) cuda::checkKernelErrors(stream, __LINE__, false)
}  // namespace cuda
