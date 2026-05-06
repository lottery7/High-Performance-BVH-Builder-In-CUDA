#include <cuda_runtime.h>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "hploc_wide.cuh"

constexpr unsigned long long invalid_wide_task = ~0ull;

__device__ __forceinline__ static BVH2Node load_bvh2_node(const BVH2Node* __restrict__ nodes, unsigned int index)
{
  const uint4* ptr = reinterpret_cast<const uint4*>(&nodes[index]);
  uint4 v0 = __ldg(&ptr[0]);
  uint4 v1 = __ldg(&ptr[1]);
  BVH2Node node;
  uint4* node_ptr = reinterpret_cast<uint4*>(&node);
  node_ptr[0] = v0;
  node_ptr[1] = v1;
  return node;
}

__device__ static int find_largest_internal_frontier_node(const BVH2Node* bvh2_nodes, const unsigned int* frontier, unsigned int frontier_size)
{
  int best_index = -1;
  float best_area = -1.0f;
  for (unsigned int i = 0; i < frontier_size; ++i) {
    const BVH2Node candidate = load_bvh2_node(bvh2_nodes, frontier[i]);
    if (candidate.is_leaf()) continue;

    const float area = candidate.aabb.half_area();
    if (area > best_area) {
      best_area = area;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

namespace cuda::hploc
{
  template <unsigned int Arity>
  __global__ void convert_to_wide_kernel(
      const BVH2Node* __restrict__ bvh2_nodes,
      WideBVHNode<Arity>* __restrict__ wide_nodes,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_wide_nodes,
      unsigned int n_faces)
  {
    static_assert(Arity == 4 || Arity == 8, "Wide H-PLOC conversion supports BVH4/BVH8 only");

    const unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= n_faces) return;

    unsigned long long task = invalid_wide_task;
    while (task == invalid_wide_task) {
      task = tasks[task_id];
    }

    while (true) {
      const unsigned int bvh2_node_index = unpack_wide_bvh2_node_index(task);
      const unsigned int wide_node_index = unpack_wide_node_index(task);

      if (wide_node_index == INVALID_INDEX) {
        return;
      }

      const BVH2Node binary_node = load_bvh2_node(bvh2_nodes, bvh2_node_index);

      if (binary_node.is_leaf()) {
        WideBVHNode<Arity> wide_node{};
        wide_node.aabb = binary_node.aabb;
        for (unsigned int i = 0; i < Arity; ++i) wide_node.children[i] = INVALID_INDEX;
        wide_node.children[1] = binary_node.right_child_index;
        wide_nodes[wide_node_index] = wide_node;
        return;
      }

      unsigned int frontier[Arity];
      frontier[0] = binary_node.left_child_index;
      frontier[1] = binary_node.right_child_index;
      unsigned int frontier_size = 2;

      while (frontier_size < Arity) {
        const int expand_index = find_largest_internal_frontier_node(bvh2_nodes, frontier, frontier_size);
        if (expand_index < 0) break;

        const BVH2Node expanded = load_bvh2_node(bvh2_nodes, frontier[expand_index]);
        frontier[expand_index] = expanded.left_child_index;
        frontier[frontier_size++] = expanded.right_child_index;
      }

      const unsigned int child_base_idx = atomicAdd(n_wide_nodes, frontier_size);
      const unsigned int new_tasks_start = atomicAdd(n_tasks, frontier_size - 1u);

      WideBVHNode<Arity> wide_node{};
      wide_node.aabb = binary_node.aabb;

      for (unsigned int i = 0; i < Arity; ++i) {
        wide_node.children[i] = INVALID_INDEX;

        if (i >= frontier_size) continue;

        const unsigned int child_idx = child_base_idx + i;
        wide_node.children[i] = child_idx;
        const unsigned long long new_task = pack_wide_task(frontier[i], child_idx);

        if (i == 0)
          task = new_task;
        else
          tasks[new_tasks_start + i - 1] = new_task;
      }

      wide_nodes[wide_node_index] = wide_node;
    }
  }

  template __global__ void convert_to_wide_kernel<4>(
      const BVH2Node* __restrict__ bvh2_nodes,
      WideBVHNode4* __restrict__ wide_nodes,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_wide_nodes,
      unsigned int n_faces);

  template __global__ void convert_to_wide_kernel<8>(
      const BVH2Node* __restrict__ bvh2_nodes,
      WideBVHNode8* __restrict__ wide_nodes,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_wide_nodes,
      unsigned int n_faces);
}  // namespace cuda::hploc
