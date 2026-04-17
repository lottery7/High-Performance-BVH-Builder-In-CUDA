#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "../structs/bvh_node.h"
#include "../structs/wide_bvh_node.h"

namespace cuda::hploc
{
  template <unsigned int Arity>
  void convert_to_wide(
      cudaStream_t stream,
      const BVHNode* d_binary_nodes,
      WideBVHNode<Arity>* d_wide_nodes,
      std::uint64_t* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int* d_block_counter,
      unsigned int n_faces);
}
