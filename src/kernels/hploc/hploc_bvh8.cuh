#pragma once

#include "../structs/bvh_node.h"

namespace cuda::hploc
{
  __host__ __device__ __forceinline__ unsigned long long pack_task(unsigned int binary_node_index, unsigned int wide_node_index)
  {
    return (static_cast<unsigned long long>(binary_node_index) << 32u) | wide_node_index;
  }

  __host__ __device__ __forceinline__ unsigned int unpack_bvh2_node_index(unsigned long long task) { return task >> 32u; }

  __host__ __device__ __forceinline__ unsigned int unpack_bvh8_node_index(unsigned long long task) { return task; }

  __global__ void build_bvh8_kernel(
      const BVH2Node* __restrict__ bvh2_nodes,
      BVH8Node* __restrict__ bvh8_nodes,
      unsigned int* __restrict__ bvh8_prim_indices,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_bvh8_nodes,
      unsigned int* __restrict__ n_bvh8_leaves,
      unsigned int n_faces);
}  // namespace cuda::hploc
