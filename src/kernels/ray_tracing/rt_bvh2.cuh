#pragma once

#include "../structs/bvh_node.h"
#include "../structs/camera.h"

namespace cuda
{
  __global__ void rt_bvh2_kernel(
      const float* vertices,
      const unsigned int* faces,
      const BVH2Node* bvh_nodes,
      unsigned int root_index,
      int* face_id,
      float* ambient_occlusion,
      const CameraView* camera,
      unsigned int n_faces);
}  // namespace cuda
