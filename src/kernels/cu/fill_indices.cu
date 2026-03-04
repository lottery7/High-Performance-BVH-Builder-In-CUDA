#include <cuda_runtime.h>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"
#include "../defines.h"

__global__ void fill_indices_kernel(unsigned int* indices, unsigned int n)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  indices[index] = index;
}

namespace cuda
{
  void fill_indices(const cudaStream_t& stream, unsigned int* indices, unsigned int n)
  {
    size_t grid = compute_grid(n);
    fill_indices_kernel<<<grid, DEFAULT_GROUP_SIZE, 0, stream>>>(indices, n);
  }
}  // namespace cuda
