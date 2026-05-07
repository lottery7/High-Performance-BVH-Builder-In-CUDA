#pragma once

#include <driver_types.h>

#include "../structs/bvh_node.h"
#include "../structs/camera.h"
#include "../structs/morton_code.h"
#include "../structs/wide_bvh_node.h"

namespace cuda
{
  template <typename T>
  void fill(cudaStream_t stream, T* arr, T val, size_t n);

  void fill_indices(cudaStream_t stream, unsigned int* d_indices, unsigned int n);

  void compute_scene_aabb(cudaStream_t stream, const AABB* __restrict__ d_aabbs, int count, AABB* __restrict__ d_scene_aabb);

  void compute_ao_radius(cudaStream_t stream, const AABB* __restrict__ d_scene_aabb, float* __restrict__ d_ao_radius);

  __global__ void compute_morton_codes_kernel(
      const AABB* __restrict__ scene_aabb,
      const unsigned int* __restrict__ faces,
      const float* __restrict__ vertices,
      MortonCode* __restrict__ morton_codes,
      unsigned int n_faces);
}  // namespace cuda
