#pragma once

#include "../structs/camera.h"
#include "../structs/wide_bvh_node.h"

namespace cuda
{
  template <unsigned int Arity>
  __global__ void rt_hploc_wide_kernel(
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode<Arity>* bvh_nodes,
      float* ambient_occlusion,
      const float* ao_radius,
      const CameraView* camera);
}  // namespace cuda
