#include <cuda_runtime.h>

#include "../../utils/cuda_utils.h"
#include "../defines.h"
#include "../structs/bvh_node_gpu.h"

__global__ void build_aabb_leaves_kernel(
    const float* vertices,
    const unsigned int* faces,
    const unsigned int* indices,
    unsigned int nfaces,
    BVHNodeGPU* lbvh)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nfaces) return;

  unsigned int i = indices[index];

  unsigned int f0 = faces[3 * i + 0];
  unsigned int f1 = faces[3 * i + 1];
  unsigned int f2 = faces[3 * i + 2];

  float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
  float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
  float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

  unsigned int leafIndex = index + nfaces - 1;

  lbvh[leafIndex].aabb.min_x = fminf(fminf(v0.x, v1.x), v2.x);
  lbvh[leafIndex].aabb.min_y = fminf(fminf(v0.y, v1.y), v2.y);
  lbvh[leafIndex].aabb.min_z = fminf(fminf(v0.z, v1.z), v2.z);
  lbvh[leafIndex].aabb.max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
  lbvh[leafIndex].aabb.max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
  lbvh[leafIndex].aabb.max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);
  lbvh[leafIndex].leftChildIndex = 0xFFFFFFFFu;
  lbvh[leafIndex].rightChildIndex = 0xFFFFFFFFu;
}

namespace cuda
{
  void build_aabb_leaves(
      const cudaStream_t& stream,
      const float* vertices,
      const unsigned int* faces,
      const unsigned int* indices,
      unsigned int nfaces,
      BVHNodeGPU* lbvh)
  {
    build_aabb_leaves_kernel<<<compute_grid(nfaces), DEFAULT_GROUP_SIZE, 0, stream>>>(vertices, faces, indices, nfaces, lbvh);
  }
}  // namespace cuda
