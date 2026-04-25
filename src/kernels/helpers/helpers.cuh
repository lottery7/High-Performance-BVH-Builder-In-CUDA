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

  template <typename KeyT, typename ValueT>
  void sort_pairs(
      cudaStream_t stream,
      const KeyT* d_keys_in,
      KeyT* d_keys_out,
      const ValueT* d_values_in,
      ValueT* d_values_out,
      int count,
      int begin_bit,
      int end_bit);

  __global__ void compute_morton_codes_kernel(
      const AABB* scene_aabb,
      unsigned int* faces,
      float* vertices,
      MortonCode* morton_codes,
      unsigned int n_faces);
}  // namespace cuda
