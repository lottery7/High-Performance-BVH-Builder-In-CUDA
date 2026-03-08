
#include "cuda_utils.h"

#include <cuda_runtime_api.h>

#include <string>

#include "exceptions.h"

namespace cuda
{
  std::string format_error(cudaError_t code) { return std::string(cudaGetErrorString(code)) + " (" + std::to_string(code) + ")"; }

  void report_error(cudaError_t err, int line, const std::string& prefix)
  {
    if (cudaSuccess == err) return;

    std::string message = prefix + format_error(err) + " at line " + std::to_string(line);

    size_t total_mem_size = 0;
    size_t free_mem_size = 0;
    cudaError_t err2;

    switch (err) {
      case cudaErrorMemoryAllocation:
        err2 = cudaMemGetInfo(&free_mem_size, &total_mem_size);
        if (cudaSuccess == err2)
          message = message + "(free memory: " + std::to_string(free_mem_size >> 20) + "/" + std::to_string(total_mem_size >> 20) + " MB)";
        else
          message = message + "(free memory unknown: " + format_error(err2) + ")";
        throw CudaBadAlloc(message);
      default:
        throw CudaException(message);
    }
  }

  void select_cuda_device(int argc, char** argv)
  {
    int device_count = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw std::runtime_error("No CUDA devices available");
    int device_id = 0;
    if (device_count > 1) {
      if (argc < 2) throw std::runtime_error("Multiple GPUs available. Pass device index as argument.");
      device_id = std::stoi(argv[1]);
      if (device_id < 0 || device_id >= device_count) throw std::runtime_error("Invalid device index");
    }
    CUDA_SAFE_CALL(cudaSetDevice(device_id));
  }
}  // namespace cuda
