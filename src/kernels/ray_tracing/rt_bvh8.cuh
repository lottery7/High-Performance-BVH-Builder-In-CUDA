#pragma once

#include "../structs/bvh_node.h"
#include "../structs/camera.h"

namespace cuda
{
  __global__ void rt_bvh8_kernel(
      const float* vertices,
      const unsigned int* faces,
      const BVH8Node* bvh_nodes,
      const unsigned int* prim_indices,
      unsigned int root_index,
      float* ambient_occlusion,
      const float* ao_radius,
      const CameraView* camera);
}  // namespace cuda
