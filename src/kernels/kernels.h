#pragma once

#include <driver_types.h>

#include "shared_structs/bvh_node_gpu_shared.h"
#include "shared_structs/camera_gpu_shared.h"

namespace cuda
{
  void aplusb(const cudaStream_t& stream, const unsigned int* a, const unsigned int* b, unsigned int* c, size_t n);

  template <typename T>
  void fill(const cudaStream_t& stream, T* arr, T val, size_t n);

  void ray_tracing_render_brute_force(
      const cudaStream_t& stream,
      dim3 gridSize,
      dim3 blockSize,
      const float* vertices,
      const unsigned int* faces,
      int* framebuffer_face_id,
      float* framebuffer_ambient_occlusion,
      CameraViewGPU* camera,
      unsigned int nfaces);

  void ray_tracing_render_using_lbvh(
      const cudaStream_t& stream,
      dim3 gridSize,
      dim3 blockSize,
      const float* vertices,
      const unsigned int* faces,
      const BVHNodeGPU* bvhNodes,
      const unsigned int* leafTriIndices,
      int* framebuffer_face_id,
      float* framebuffer_ambient_occlusion,
      CameraViewGPU* camera,
      unsigned int nfaces);
}  // namespace cuda
