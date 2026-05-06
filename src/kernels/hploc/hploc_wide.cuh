#pragma once

#include "../structs/bvh_node.h"
#include "../structs/wide_bvh_node.h"

namespace cuda::hploc
{
  __host__ __device__ __forceinline__ unsigned long long pack_wide_task(unsigned int binary_node_index, unsigned int wide_node_index)
  {
    return (static_cast<unsigned long long>(binary_node_index) << 32u) | wide_node_index;
  }

  __host__ __device__ __forceinline__ unsigned int unpack_wide_bvh2_node_index(unsigned long long task) { return task >> 32u; }

  __host__ __device__ __forceinline__ unsigned int unpack_wide_node_index(unsigned long long task) { return task; }

  template <unsigned int Arity>
  __global__ void convert_to_wide_kernel(
      const BVH2Node* __restrict__ bvh2_nodes,
      WideBVHNode<Arity>* __restrict__ wide_nodes,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_wide_nodes,
      unsigned int n_faces);
}  // namespace cuda::hploc
