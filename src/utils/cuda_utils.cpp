
#include "cuda_utils.h"

#include <cuda_runtime_api.h>

#include <string>

#include "exceptions.h"

#define CUDA_KERNELS_ACCURATE_ERRORS_CHECKS true

namespace cuda
{

  std::string formatError(cudaError_t code) { return std::string(cudaGetErrorString(code)) + " (" + std::to_string(code) + ")"; }

  void reportError(cudaError_t err, int line, const std::string& prefix)
  {
    if (cudaSuccess == err) return;

    std::string message = prefix + formatError(err) + " at line " + std::to_string(line);

    size_t total_mem_size = 0;
    size_t free_mem_size = 0;
    cudaError_t err2;

    switch (err) {
      case cudaErrorMemoryAllocation:
        err2 = cudaMemGetInfo(&free_mem_size, &total_mem_size);
        if (cudaSuccess == err2)
          message = message + "(free memory: " + std::to_string(free_mem_size >> 20) + "/" + std::to_string(total_mem_size >> 20) + " MB)";
        else
          message = message + "(free memory unknown: " + formatError(err2) + ")";
        throw cuda_bad_alloc(message);
      default:
        throw cuda_exception(message);
    }
  }

  void checkKernelErrors(cudaStream_t stream, int line, bool synchronized)
  {
    reportError(cudaGetLastError(), line, "Kernel failed: ");
    if (synchronized) {
      reportError(cudaStreamSynchronize(stream), line, "Kernel failed: ");
    }
  }

  void checkKernelErrors(cudaStream_t stream, int line) { checkKernelErrors(stream, line, CUDA_KERNELS_ACCURATE_ERRORS_CHECKS); }

  void selectCudaDevice(int argc, char** argv)
  {
    int deviceCount = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) throw std::runtime_error("No CUDA devices available");
    int deviceId = 0;
    if (deviceCount > 1) {
      if (argc < 2) throw std::runtime_error("Multiple GPUs available. Pass device index as argument.");
      deviceId = std::stoi(argv[1]);
      if (deviceId < 0 || deviceId >= deviceCount) throw std::runtime_error("Invalid device index");
    }
    CUDA_SAFE_CALL(cudaSetDevice(deviceId));
  }

}  // namespace cuda
