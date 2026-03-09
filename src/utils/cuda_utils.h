#pragma once

#include <driver_types.h>

#include <string>

#define CUDA_SAFE_CALL(expr) cuda::report_error(expr, __LINE__)
#define CUDA_SYNC_STREAM(expr) cuda::report_error(cudaStreamSynchronize(expr), __LINE__)

namespace cuda
{
  ::std::string format_error(cudaError_t code);

  void report_error(cudaError_t err, int line, const ::std::string& prefix = ::std::string());

  void select_cuda_device(int argc, char** argv);
}  // namespace cuda
