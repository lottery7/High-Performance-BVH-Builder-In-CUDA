#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "../structs/bvh_node.h"
#include "../structs/wide_bvh_node.h"

namespace cuda::wide_hploc
{
  void convert(
      cudaStream_t stream,
      const BVHNode* d_bvh2,
      WideBVHNode* d_wide_bvh,
      unsigned int* d_node_count,
      uint64_t* d_work_items,
      unsigned int* d_work_counter,
      unsigned int* d_work_alloc_counter,
      unsigned int n_bvh2_nodes,
      unsigned int n_faces);
}  // namespace cuda::wide_hploc
