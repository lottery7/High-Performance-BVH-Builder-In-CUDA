#pragma once

#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

namespace cuda::hploc
{
  __global__ void build_leaves_nodes_kernel(const unsigned int* faces, unsigned int n_faces, const float* vertices, BVHNode* nodes);

  __global__ void build_kernel(
      unsigned int* parents,
      const MortonCode* morton_codes,
      BVHNode* nodes,
      unsigned int* cluster_ids,
      unsigned int* n_clusters,
      unsigned int n_faces);
}  // namespace cuda::hploc
