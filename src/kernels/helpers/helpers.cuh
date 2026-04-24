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

  __global__ void compute_mortons_kernel(AABB scene_aabb, unsigned int* faces, float* vertices, MortonCode* morton_codes, unsigned int n_faces);
}  // namespace cuda
