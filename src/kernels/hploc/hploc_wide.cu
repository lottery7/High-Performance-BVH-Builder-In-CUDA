#include <cuda_runtime.h>

#include <cstdint>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "hploc_wide.cuh"

#define INVALID_TASK (~0ull)

template <unsigned int Arity>
__device__ void static initialize_wide_node(WideBVHNode<Arity>& node)
{
  node.valid_mask = 0;
  node.primitive_mask = 0;
  for (unsigned int i = 0; i < Arity; ++i) {
    node.child_indices[i] = INVALID_INDEX;
    node.child_aabbs[i] = node.aabb;
  }
}

__device__ static int find_largest_internal_frontier_node(const BVH2Node* binary_nodes, const unsigned int* frontier, unsigned int frontier_size)
{
  int best_index = -1;
  float best_area = -1.0f;
  for (unsigned int i = 0; i < frontier_size; ++i) {
    const BVH2Node& candidate = binary_nodes[frontier[i]];
    if (candidate.is_leaf()) continue;
    const float area = candidate.aabb.surface_area();
    if (area > best_area) {
      best_area = area;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

__host__ __device__ inline float3 operator-(float3 lhs, float3 rhs)
{
  return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__device__ __forceinline__ std::uint32_t ceil_log2(float x)
{
  const std::uint32_t ix = __float_as_uint(x);
  const std::uint32_t exp = (ix >> 23) & 0xFFu;
  const bool is_pow2 = (ix & ((1u << 23) - 1u)) == 0u;
  return exp + !is_pow2;
}

__device__ __forceinline__ float inv_pow2(std::uint8_t e_biased)
{
  return __uint_as_float(static_cast<std::uint32_t>(254u - e_biased) << 23);
}

__device__ cuda::hploc::BVH8Node create_compact_bvh8_node(
    const BVH2Node* binary_nodes,
    const BVH2Node& binary_node,
    const unsigned int* frontier,
    unsigned int frontier_size,
    unsigned int child_base_index,
    unsigned int prim_base_index)
{
  cuda::hploc::BVH8NodeExplicit node{};
  node.p = make_float3(binary_node.aabb.min_x, binary_node.aabb.min_y, binary_node.aabb.min_z);

  const float3 diagonal = make_float3(
      binary_node.aabb.max_x - binary_node.aabb.min_x,
      binary_node.aabb.max_y - binary_node.aabb.min_y,
      binary_node.aabb.max_z - binary_node.aabb.min_z);
  constexpr float quant_step = 1.0f / 255.0f;
  node.e[0] = static_cast<std::uint8_t>(ceil_log2(diagonal.x * quant_step));
  node.e[1] = static_cast<std::uint8_t>(ceil_log2(diagonal.y * quant_step));
  node.e[2] = static_cast<std::uint8_t>(ceil_log2(diagonal.z * quant_step));
  node.childBaseIdx = child_base_index;
  node.primBaseIdx = prim_base_index;

  const float3 inv_e = make_float3(inv_pow2(node.e[0]), inv_pow2(node.e[1]), inv_pow2(node.e[2]));
  unsigned int next_internal_offset = 0;
  unsigned int next_leaf_offset = 0;
  for (unsigned int slot = 0; slot < frontier_size; ++slot) {
    const BVH2Node& child = binary_nodes[frontier[slot]];
    if (child.is_leaf()) {
      node.meta[slot] = static_cast<std::uint8_t>((1u << 5) | next_leaf_offset++);
    } else {
      node.imask |= 1u << slot;
      node.meta[slot] = static_cast<std::uint8_t>((1u << 5) | (24u + next_internal_offset++));
    }

    node.qlox[slot] = static_cast<std::uint8_t>(floorf((child.aabb.min_x - binary_node.aabb.min_x) * inv_e.x));
    node.qloy[slot] = static_cast<std::uint8_t>(floorf((child.aabb.min_y - binary_node.aabb.min_y) * inv_e.y));
    node.qloz[slot] = static_cast<std::uint8_t>(floorf((child.aabb.min_z - binary_node.aabb.min_z) * inv_e.z));
    node.qhix[slot] = static_cast<std::uint8_t>(ceilf((child.aabb.max_x - binary_node.aabb.min_x) * inv_e.x));
    node.qhiy[slot] = static_cast<std::uint8_t>(ceilf((child.aabb.max_y - binary_node.aabb.min_y) * inv_e.y));
    node.qhiz[slot] = static_cast<std::uint8_t>(ceilf((child.aabb.max_z - binary_node.aabb.min_z) * inv_e.z));
  }

  return *reinterpret_cast<cuda::hploc::BVH8Node*>(&node);
}

namespace cuda::hploc
{
  template <unsigned int Arity>
  __global__ void convert_to_wide_kernel(
      const BVH2Node* binary_nodes,
      WideBVHNode<Arity>* wide_nodes,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_wide_node,
      unsigned int n_faces)
  {
    const unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= n_faces) return;

    // Busy waiting тасок воркерами
    unsigned long long task = INVALID_TASK;
    while (task == INVALID_TASK) {
      task = tasks[task_id];
    }

    // Одну из тасок передаем сами себе для экономии
    while (true) {
      const unsigned int binary_node_index = unpack_binary_node_index(task);
      const unsigned int wide_node_index = unpack_wide_node_index(task);

      if (wide_node_index == INVALID_INDEX) {
        return;
      }

      const BVH2Node& binary_node = binary_nodes[binary_node_index];

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

        const BVH2Node& expanded = binary_nodes[frontier[expand_index]];
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
        const BVH2Node& child = binary_nodes[frontier[i]];
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
          tasks[new_tasks_start + i - 1] = new_task;
        }
      }

      wide_nodes[wide_node_index] = wide_node;
    }
  }

  template <unsigned int Arity>
  void convert_to_wide(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
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

  __global__ void convert_to_bvh8_kernel(
      const BVH2Node* binary_nodes,
      BVH8Node* bvh8_nodes,
      unsigned int* prim_indices,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_bvh8_node,
      unsigned int* next_prim,
      unsigned int n_faces)
  {
    const unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= n_faces) return;

    unsigned long long task = INVALID_TASK;
    while (task == INVALID_TASK) {
      task = tasks[task_id];
    }

    while (true) {
      const unsigned int binary_node_index = unpack_binary_node_index(task);
      const unsigned int bvh8_node_index = unpack_wide_node_index(task);

      if (bvh8_node_index == INVALID_INDEX) return;

      const BVH2Node& binary_node = binary_nodes[binary_node_index];

      if (binary_node.is_leaf()) {
        const unsigned int prim_base_index = atomicAdd(next_prim, 1u);
        prim_indices[prim_base_index] = binary_node.right_child_index;

        const unsigned int frontier[8] = {binary_node_index, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        bvh8_nodes[bvh8_node_index] = create_compact_bvh8_node(binary_nodes, binary_node, frontier, 1u, 0u, prim_base_index);
        return;
      }

      unsigned int frontier[8];
      frontier[0] = binary_node.left_child_index;
      frontier[1] = binary_node.right_child_index;
      unsigned int frontier_size = 2;

      while (frontier_size < 8) {
        const int expand_index = find_largest_internal_frontier_node(binary_nodes, frontier, frontier_size);
        if (expand_index < 0) break;

        const BVH2Node& expanded = binary_nodes[frontier[expand_index]];
        frontier[expand_index] = expanded.left_child_index;
        frontier[frontier_size++] = expanded.right_child_index;
      }

      unsigned int n_internal_children = 0;
      unsigned int n_leaf_children = 0;
      for (unsigned int i = 0; i < frontier_size; ++i) {
        if (binary_nodes[frontier[i]].is_leaf())
          ++n_leaf_children;
        else
          ++n_internal_children;
      }

      const unsigned int child_base_index = atomicAdd(next_bvh8_node, n_internal_children);
      const unsigned int prim_base_index = n_leaf_children == 0 ? 0u : atomicAdd(next_prim, n_leaf_children);
      const unsigned int new_tasks_start = atomicAdd(next_task, frontier_size - 1u);

      unsigned int next_internal_offset = 0;
      unsigned int next_leaf_offset = 0;
      for (unsigned int i = 0; i < frontier_size; ++i) {
        const BVH2Node& child = binary_nodes[frontier[i]];
        unsigned long long new_task = INVALID_TASK;
        if (child.is_leaf()) {
          prim_indices[prim_base_index + next_leaf_offset++] = child.right_child_index;
          new_task = pack_task(frontier[i], INVALID_INDEX);
        } else {
          const unsigned int child_index = child_base_index + next_internal_offset++;
          new_task = pack_task(frontier[i], child_index);
        }

        if (i == 0) {
          task = new_task;
        } else {
          tasks[new_tasks_start + i - 1u] = new_task;
        }
      }

      bvh8_nodes[bvh8_node_index] = create_compact_bvh8_node(binary_nodes, binary_node, frontier, frontier_size, child_base_index, prim_base_index);
    }
  }

  void convert_to_bvh8(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
      BVH8Node* d_bvh8_nodes,
      unsigned int* d_prim_indices,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_bvh8_node,
      unsigned int* d_next_prim,
      unsigned int n_faces)
  {
    rassert(n_faces > 0, 61824858);

    const unsigned int root_index = 2 * n_faces - 2;
    const unsigned long long root_task = pack_task(root_index, 0);
    const unsigned int zero = 0;
    const unsigned int one = 1;

    CUDA_SAFE_CALL(cudaMemsetAsync(d_tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_tasks, &root_task, sizeof(root_task), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_task, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_bvh8_node, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_prim, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));

    convert_to_bvh8_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        d_binary_nodes,
        d_bvh8_nodes,
        d_prim_indices,
        reinterpret_cast<unsigned long long*>(d_tasks),
        d_next_task,
        d_next_bvh8_node,
        d_next_prim,
        n_faces);
  }

  template void convert_to_wide<4>(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
      WideBVHNode4* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);

  template void convert_to_wide<8>(
      cudaStream_t stream,
      const BVH2Node* d_binary_nodes,
      WideBVHNode8* d_wide_nodes,
      unsigned long long* d_tasks,
      unsigned int* d_next_task,
      unsigned int* d_next_wide_node,
      unsigned int n_faces);

  template __global__ void convert_to_wide_kernel<4>(
      const BVH2Node* binary_nodes,
      WideBVHNode<4>* wide_nodes,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_wide_node,
      unsigned int n_faces);

  template __global__ void convert_to_wide_kernel<8>(
      const BVH2Node* binary_nodes,
      WideBVHNode<8>* wide_nodes,
      volatile unsigned long long* tasks,
      unsigned int* next_task,
      unsigned int* next_wide_node,
      unsigned int n_faces);

}  // namespace cuda::hploc
