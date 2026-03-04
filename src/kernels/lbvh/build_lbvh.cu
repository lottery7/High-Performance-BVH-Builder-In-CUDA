#include <cuda_runtime.h>

#include "../../utils/cuda_utils.h"
#include "../defines.h"
#include "../structs/bvh_node_gpu.h"

__device__ __forceinline__ static int common_bits_from(const unsigned int* arr, int n, int i, int j)
{
  if (i < 0 || j < 0 || i >= n || j >= n) return -1;
  if (arr[i] == arr[j]) return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
  return __clz(static_cast<unsigned int>(arr[i]) ^ static_cast<unsigned int>(arr[j]));
}

__global__ void build_lbvh_kernel(const unsigned int* morton_codes, int nfaces, BVHNodeGPU* lbvh)
{
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index > nfaces - 2) return;

  int diff = common_bits_from(morton_codes, nfaces, index, index + 1) - common_bits_from(morton_codes, nfaces, index, index - 1);

  int direction = (diff > 0) ? 1 : -1;
  int dmin = common_bits_from(morton_codes, nfaces, index, index - direction);

  int lmax = 2;
  while (common_bits_from(morton_codes, nfaces, index, index + lmax * direction) > dmin) lmax *= 2;

  int l = 0;
  for (int t = lmax / 2; t > 0; t /= 2) {
    if (common_bits_from(morton_codes, nfaces, index, index + (l + t) * direction) > dmin) l += t;
  }

  int j = index + l * direction;
  int dnode = common_bits_from(morton_codes, nfaces, index, j);

  int s = 0;
  for (int t = lmax >> 1; t > 0; t >>= 1) {
    if (common_bits_from(morton_codes, nfaces, index, index + (s + t) * direction) > dnode) s += t;
  }

  int y = index + s * direction + min(direction, 0);

  int min_val = min(index, j);
  int max_val = max(index, j);

  lbvh[index].leftChildIndex = (min_val == y) ? nfaces - 1 + min_val : y;
  lbvh[index].rightChildIndex = (max_val == y + 1) ? nfaces - 1 + max_val : y + 1;
}

namespace cuda
{
  void build_lbvh(const cudaStream_t& stream, const unsigned int* morton_codes, unsigned int nfaces, BVHNodeGPU* lbvh)
  {
    if (nfaces < 2) return;
    build_lbvh_kernel<<<compute_grid(nfaces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(morton_codes, static_cast<int>(nfaces), lbvh);
  }
}  // namespace cuda
