#include <cuda_runtime.h>

#include <cstdio>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"
#include "../defines.h"

template <typename T>
__global__ void fill_kernel(T* arr, T val, size_t n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) arr[i] = val;
}

namespace cuda
{
  template <typename T>
  void fill(const cudaStream_t& stream, T* arr, T val, size_t n)
  {
    size_t grid = compute_grid(n);
    fill_kernel<<<grid, DEFAULT_GROUP_SIZE, 0, stream>>>(arr, val, n);
  }

  template void fill(const cudaStream_t& stream, unsigned int* arr, unsigned int val, size_t n);
  template void fill(const cudaStream_t& stream, int* arr, int val, size_t n);
  template void fill(const cudaStream_t& stream, float* arr, float val, size_t n);
}  // namespace cuda