#include <cuda_runtime.h>

#include <cstdint>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "hploc_wide.h"

#define INVALID_TASK (~0ull)

__host__ __device__ __forceinline__ unsigned long long pack_task(unsigned int binary_node_index, unsigned int wide_node_index)
{
  return (static_cast<unsigned long long>(binary_node_index) << 32u) | static_cast<unsigned long long>(wide_node_index);
}

__host__ __device__ __forceinline__ unsigned int unpack_binary_node_index(unsigned long long task) { return static_cast<unsigned int>(task >> 32u); }

__host__ __device__ __forceinline__ unsigned int unpack_wide_node_index(unsigned long long task) { return static_cast<unsigned int>(task); }

template <unsigned int Arity>
__device__ void initialize_wide_node(WideBVHNode<Arity>& node)
{
  node.valid_mask = 0;
  node.primitive_mask = 0;
  for (unsigned int i = 0; i < Arity; ++i) {
    node.child_indices[i] = INVALID_INDEX;
    node.child_aabbs[i] = node.aabb;
  }
}

__device__ int find_largest_internal_frontier_node(const BVHNode* binary_nodes, const unsigned int* frontier, unsigned int frontier_size)
{
  int best_index = -1;
  float best_area = -1.0f;
  for (unsigned int i = 0; i < frontier_size; ++i) {
    const BVHNode& candidate = binary_nodes[frontier[i]];
    if (candidate.is_leaf()) continue;
    const float area = candidate.aabb.surface_area();
    if (area > best_area) {
      best_area = area;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

template <unsigned int Arity>
__global__ void convert_to_wide_kernel(
    const BVHNode* binary_nodes,
    WideBVHNode<Arity>* wide_nodes,
    unsigned long long* tasks,
    unsigned int* next_task,
    unsigned int* next_wide_node,
    unsigned int n_faces)
{
  const unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (task_id >= n_faces) return;

  auto* volatile_tasks = reinterpret_cast<volatile unsigned long long*>(tasks);

  // Busy waiting тасок воркерами
  unsigned long long task = INVALID_TASK;
  while (task == INVALID_TASK) {
    task = volatile_tasks[task_id];
  }

  // Одну из тасок передаем сами себе для экономии
  while (true) {
    const unsigned int binary_node_index = unpack_binary_node_index(task);
    const unsigned int wide_node_index = unpack_wide_node_index(task);

    if (wide_node_index == INVALID_INDEX) {
      // Убрал, чтобы избежать лишних n_faces uncoalesced чтений из памяти
      // curassert(binary_node.is_leaf(), 24592011);
      return;
    }

    const BVHNode& binary_node = binary_nodes[binary_node_index];

    // Если у нас больше одного примитива, то корень не может быть листом. Значит таску положил другой воркер.
    // Лист мы кладем только с wide_node_index = INVALID_INDEX, а этот случай уже был пройден выше.
    curassert(!binary_node.is_leaf(), 24341771);

    WideBVHNode<Arity> wide_node{};
    wide_node.aabb = binary_node.aabb;
    initialize_wide_node(wide_node);

    // Ищем своих детей: расширяем бинарную ноду, записываем индексы во frontier
    unsigned int frontier[Arity];
    frontier[0] = binary_node.left_child_index;
    frontier[1] = binary_node.right_child_index;
    unsigned int frontier_size = 2;

    while (frontier_size < Arity) {
      const int expand_index = find_largest_internal_frontier_node(binary_nodes, frontier, frontier_size);
      if (expand_index < 0) break;

      const BVHNode& expanded = binary_nodes[frontier[expand_index]];
      frontier[expand_index] = expanded.left_child_index;
      frontier[frontier_size++] = expanded.right_child_index;
    }

    unsigned int n_internal_children = 0;
    for (unsigned int i = 0; i < frontier_size; ++i) {
      if (binary_nodes[frontier[i]].left_child_index != INVALID_INDEX) ++n_internal_children;
    }

    // Аллоцируем место под n_internal_children новых широких нод
    const unsigned int child_base_index = atomicAdd(next_wide_node, n_internal_children);
    wide_node.valid_mask = (1u << frontier_size) - 1u;

    const unsigned int new_tasks_start = atomicAdd(next_task, frontier_size - 1u);
    unsigned int next_internal_offset = 0;

    for (unsigned int i = 0; i < frontier_size; ++i) {
      const BVHNode& child = binary_nodes[frontier[i]];
      wide_node.child_aabbs[i] = child.aabb;
      unsigned long long new_task = INVALID_TASK;

      if (child.is_leaf()) {
        wide_node.primitive_mask |= 1u << i;
        wide_node.child_indices[i] = child.right_child_index;
        new_task = pack_task(frontier[i], INVALID_INDEX);
      } else {
        wide_node.child_indices[i] = child_base_index + next_internal_offset++;
        new_task = pack_task(frontier[i], wide_node.child_indices[i]);
      }

      if (i == 0) {
        task = new_task;
      } else {
        atomicExch(&tasks[new_tasks_start + i - 1], new_task);
      }
    }
    __threadfence();
    wide_nodes[wide_node_index] = wide_node;
  }
}

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
      unsigned int n_faces)
  {
    rassert(n_faces > 1, 61824857);

    const unsigned int root_index = 2 * n_faces - 2;
    const unsigned long long root_task = pack_task(root_index, 0);
    const unsigned int one = 1;

    CUDA_SAFE_CALL(cudaMemsetAsync(d_tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_tasks, &root_task, sizeof(root_task), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_task, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_wide_node, &one, sizeof(one), cudaMemcpyHostToDevice, stream));

    convert_to_wide_kernel<Arity><<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        d_binary_nodes,
        d_wide_nodes,
        reinterpret_cast<unsigned long long*>(d_tasks),
        d_next_task,
        d_next_wide_node,
        n_faces);
  }

  template void convert_to_wide<4>(
      cudaStream_t stream,
      const BVHNode* d_binary_nodes,
      WideBVHNode4* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);

  template void convert_to_wide<8>(
      cudaStream_t stream,
      const BVHNode* d_binary_nodes,
      WideBVHNode8* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);
}  // namespace cuda::hploc
