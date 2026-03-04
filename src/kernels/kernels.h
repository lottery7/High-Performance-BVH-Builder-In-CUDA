#pragma once

#include <driver_types.h>

#include "shared_structs/bvh_node_gpu_shared.h"
#include "shared_structs/camera_gpu_shared.h"

namespace cuda
{
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

  void ray_tracing_render_using_bvh(
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

  void fill_indices(const cudaStream_t& stream, unsigned int* indices, unsigned int n);

  void compute_mortons(
      const cudaStream_t& stream,
      const unsigned int* faces,
      const float* vertices,
      unsigned int nfaces,
      float cMinX,
      float cMinY,
      float cMinZ,
      float cMaxX,
      float cMaxY,
      float cMaxZ,
      unsigned int* morton_codes);

  void build_lbvh(const cudaStream_t& stream, const unsigned int* morton_codes, unsigned int nfaces, BVHNodeGPU* lbvh);

  void build_aabb_leaves(
      const cudaStream_t& stream,
      const float* vertices,
      const unsigned int* faces,
      const unsigned int* indices,
      unsigned int nfaces,
      BVHNodeGPU* lbvh);

  void build_aabb(const cudaStream_t& stream, unsigned int nfaces, BVHNodeGPU* lbvh, int* parent, unsigned int* flags);

  template <typename K, typename V>
  void sort_by_key(const cudaStream_t& stream, K* d_keys, V* d_vals, size_t n);
}  // namespace cuda
