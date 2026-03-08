#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../defines.h"
#include "../structs/bvh_node.h"

__device__ __forceinline__ static int common_bits_from(const unsigned int* arr, int n, int i, int j)
{
  if (i < 0 || j < 0 || i >= n || j >= n) return -1;
  if (arr[i] == arr[j]) return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
  return __clz(static_cast<unsigned int>(arr[i]) ^ static_cast<unsigned int>(arr[j]));
}

__global__ void build_lbvh_kernel(const unsigned int* morton_codes, int nfaces, BVHNode* lbvh)
{
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index > nfaces - 2) return;

  int diff = common_bits_from(morton_codes, nfaces, index, index + 1) - common_bits_from(morton_codes, nfaces, index, index - 1);

  int direction = (diff > 0) ? 1 : -1;
  int dmin = common_bits_from(morton_codes, nfaces, index, index - direction);

  int r_max = 2;
  while (common_bits_from(morton_codes, nfaces, index, index + r_max * direction) > dmin) r_max *= 2;

  int r = 0;
  for (int t = r_max / 2; t > 0; t /= 2) {
    if (common_bits_from(morton_codes, nfaces, index, index + (r + t) * direction) > dmin) r += t;
  }

  r = index + r * direction;
  int dnode = common_bits_from(morton_codes, nfaces, index, r);

  int s = 0;
  for (int t = r_max >> 1; t > 0; t >>= 1) {
    if (common_bits_from(morton_codes, nfaces, index, index + (s + t) * direction) > dnode) s += t;
  }

  int y = index + s * direction + min(direction, 0);

  int min_val = min(index, r);
  int max_val = max(index, r);

  lbvh[index].left_child_index = (min_val == y) ? nfaces - 1 + min_val : y;
  lbvh[index].right_child_index = (max_val == y + 1) ? nfaces - 1 + max_val : y + 1;
}

namespace cuda
{
  void build_lbvh(cudaStream_t stream, const unsigned int* d_morton_codes, unsigned int n_faces, BVHNode* d_lbvh)
  {
    rassert(n_faces > 2, 638420165);
    build_lbvh_kernel<<<compute_grid(n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(d_morton_codes, static_cast<int>(n_faces), d_lbvh);
  }
}  // namespace cuda
