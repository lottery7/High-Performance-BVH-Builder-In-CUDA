#pragma once

#include "../structs/camera.h"
#include "kernels/nexus_bvh/nexus_bvh8.cuh"

namespace cuda
{
  __global__ void rt_compressed_bvh8_kernel(
      const float* vertices,
      const unsigned int* faces,
      const nexus_bvh_wide::BVH8Node* bvh_nodes,
      const unsigned int* prim_idx,
      int* face_id,
      float* ambient_occlusion,
      const CameraView* camera);
}  // namespace cuda
