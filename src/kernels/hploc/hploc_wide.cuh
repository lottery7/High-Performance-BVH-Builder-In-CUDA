#pragma once

#include <cuda_runtime.h>

#include "../nexus_bvh/nexus_bvh8.cuh"
#include "../structs/bvh_node.h"
#include "../structs/wide_bvh_node.h"

namespace cuda::hploc
{
  using BVH8NodeExplicit = cuda::nexus_bvh_wide::BVH8NodeExplicit;
  using BVH8Node = cuda::nexus_bvh_wide::BVH8Node;

  template <unsigned int Arity>
  void convert_to_wide(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
      WideBVHNode<Arity>* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);

  void convert_to_bvh8(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
      BVH8Node* d_bvh8_nodes,
      unsigned int* d_prim_indices,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_bvh8_node,
      unsigned int* d_next_prim,
      unsigned int n_faces);

  template <unsigned int Arity>
  __global__ void convert_to_wide_kernel(
      const BVH2Node* binary_nodes,
      WideBVHNode<Arity>* wide_nodes,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_wide_node,
      unsigned int n_faces);

  __global__ void convert_to_bvh8_kernel(
      const BVH2Node* binary_nodes,
      BVH8Node* bvh8_nodes,
      unsigned int* prim_indices,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_bvh8_node,
      unsigned int* next_prim,
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
