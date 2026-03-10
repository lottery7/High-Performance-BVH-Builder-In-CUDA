#pragma once

#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

namespace cuda::hploc
{
  void build(
      cudaStream_t stream,
      AABB scene_aabb,
      unsigned int* d_faces,
      float* d_vertices,
      BVHNode* d_nodes,
      unsigned int* d_parents,
      MortonCode* d_morton_codes,
      unsigned int* d_cluster_ids,
      unsigned int* d_n_clusters,
      unsigned int n_faces);
}