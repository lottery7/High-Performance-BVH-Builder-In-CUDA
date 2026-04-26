#pragma once

#include "../structs/bvh_node.h"

namespace cuda::hploc
{
  __global__ void build_leaves_kernel(
      const unsigned int* __restrict__ faces,
      unsigned int n_faces,
      const float* __restrict__ vertices,
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ clusters,
      AABB* __restrict__ scene_aabb);

  __global__ void build_kernel(
      unsigned int* __restrict__ parents,
      const MortonCode* __restrict__ morton_codes,
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ cluster_ids,
      unsigned int* __restrict__ n_clusters,
      unsigned int n_faces);
}  // namespace cuda::hploc
