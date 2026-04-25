#pragma once

#include "../structs/bvh_node.h"

namespace cuda::hploc_bvh8
{
  __global__ void build_leaves_kernel(
      const unsigned int* __restrict__ faces,
      unsigned int n_faces,
      const float* __restrict__ vertices,
      BVH2Node* __restrict__ nodes,
      AABB* __restrict__ primitives_aabb,
      unsigned int* __restrict__ clusters);

  __global__ void build_kernel(
      unsigned int* parents,
      const MortonCode* morton_codes,
      BVH2Node* nodes,
      unsigned int* cluster_ids,
      unsigned int* n_clusters,
      unsigned int n_faces);
}  // namespace cuda::hploc
