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
}
