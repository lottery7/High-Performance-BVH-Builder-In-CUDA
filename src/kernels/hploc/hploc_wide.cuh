#pragma once

#include <cuda_runtime.h>

#include "../structs/bvh_node.h"
#include "../structs/wide_bvh_node.h"

namespace cuda::hploc
{
  template <unsigned int Arity>
  void convert_to_wide(
      cudaStream_t stream,
      const BVHNode* d_binary_nodes,
      WideBVHNode<Arity>* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);

  template <unsigned int Arity>
  __global__ void convert_to_wide_kernel(
      const BVHNode* binary_nodes,
      WideBVHNode<Arity>* wide_nodes,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_wide_node,
      unsigned int n_faces);

  __host__ __device__ __forceinline__ unsigned long long pack_task(unsigned int binary_node_index, unsigned int wide_node_index)
  {
    return (static_cast<unsigned long long>(binary_node_index) << 32u) | static_cast<unsigned long long>(wide_node_index);
  }

  __host__ __device__ __forceinline__ unsigned int unpack_binary_node_index(unsigned long long task)
  {
    return static_cast<unsigned int>(task >> 32u);
  }

  __host__ __device__ __forceinline__ unsigned int unpack_wide_node_index(unsigned long long task) { return static_cast<unsigned int>(task); }
}  // namespace cuda::hploc
