#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../defines.h"
#include "../structs/bvh_node.h"

__global__ void build_aabb_leaves_kernel(
    const float* vertices,
    const unsigned int* faces,
    const unsigned int* indices,
    unsigned int n_faces,
    BVHNode* lbvh)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_faces) return;

  unsigned int i = indices[index];

  unsigned int f0 = faces[3 * i + 0];
  unsigned int f1 = faces[3 * i + 1];
  unsigned int f2 = faces[3 * i + 2];

  float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
  float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
  float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

  unsigned int leaf_index = index + n_faces - 1;

  lbvh[leaf_index].aabb.min_x = fminf(fminf(v0.x, v1.x), v2.x);
  lbvh[leaf_index].aabb.min_y = fminf(fminf(v0.y, v1.y), v2.y);
  lbvh[leaf_index].aabb.min_z = fminf(fminf(v0.z, v1.z), v2.z);
  lbvh[leaf_index].aabb.max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
  lbvh[leaf_index].aabb.max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
  lbvh[leaf_index].aabb.max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);
  lbvh[leaf_index].left_child_index = 0xFFFFFFFFu;
  lbvh[leaf_index].right_child_index = 0xFFFFFFFFu;
}

namespace cuda
{
  void build_aabb_leaves(
      cudaStream_t stream,
      const float* d_vertices,
      const unsigned int* d_faces,
      const unsigned int* d_indices,
      unsigned int n_faces,
      BVHNode* d_lbvh)
  {
    build_aabb_leaves_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(d_vertices, d_faces, d_indices, n_faces, d_lbvh);
  }
}  // namespace cuda
