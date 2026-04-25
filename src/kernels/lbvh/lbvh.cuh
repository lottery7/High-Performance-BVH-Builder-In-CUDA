#pragma once

#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

namespace cuda::lbvh
{
  __global__ void build_bvh_kernel(
      BVH2Node* __restrict__ nodes,
      MortonCode* __restrict__ morton_codes,
      unsigned int* __restrict__ parents,
      const AABB* __restrict__ primitives_aabb,
      const unsigned int* __restrict__ primitives,
      unsigned int n_faces);

  __global__ void build_primitives_aabb_kernel(
      const unsigned int* __restrict__ faces,
      unsigned int n_faces,
      const float* __restrict__ vertices,
      unsigned int* __restrict__ primitive_indices,
      AABB* __restrict__ primitives_aabb);

  __global__ void build_internal_nodes_aabb_kernel(
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ parents,
      unsigned int* __restrict__ flags,
      unsigned int n_faces);
}  // namespace cuda::lbvh
