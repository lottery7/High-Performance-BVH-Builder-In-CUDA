#pragma once

#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"
#include "../structs/scene.h"

namespace cuda::lbvh
{
  void build(
      cudaStream_t stream,
      AABB scene_aabb,
      unsigned int *d_faces,
      float *d_vertices,
      BVHNode *d_bvh,
      unsigned int *d_morton_codes,
      unsigned int *d_indices,
      unsigned int *d_parents,
      unsigned int *d_flags,
      unsigned int n_faces);
}  // namespace cuda::lbvh
