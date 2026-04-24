#pragma once

#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

namespace cuda::lbvh
{
  __global__ void build_bvh_kernel(BVHNode* bvh, MortonCode* morton_codes, unsigned int n_faces);

  __global__ void build_aabb_leaves_kernel(BVHNode* bvh, unsigned int* faces, float* vertices, unsigned int* indices, unsigned int n_faces);

  __global__ void build_aabb_kernel(BVHNode* bvh, unsigned int* parents, unsigned int* flags, unsigned int n_faces);

  __global__ void compute_parents_kernel(BVHNode* bvh, unsigned int* parents, unsigned int n_faces);
}  // namespace cuda::lbvh
