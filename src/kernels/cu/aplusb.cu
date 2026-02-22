#include <cuda_runtime.h>

#include <cstdio>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"
#include "../defines.h"

__global__ void aplusb_kernel(const unsigned int* a, const unsigned int* b, unsigned int* c, size_t n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = index; i < n; i += stride) c[i] = a[i] + b[i];
}

namespace cuda
{
  void aplusb(const cudaStream_t& stream, const unsigned int* a, const unsigned int* b, unsigned int* c, size_t n)
  {
    constexpr int block = DEFAULT_GROUP_SIZE;
    int grid = std::min(divCeil(n, block), 65536);
    aplusb_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
  }
}  // namespace cuda
