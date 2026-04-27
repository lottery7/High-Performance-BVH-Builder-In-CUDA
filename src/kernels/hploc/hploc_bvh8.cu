#include <cooperative_groups/details/helpers.h>
#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../helpers/helpers.cuh"
#include "../structs/aabb.h"
#include "hploc_bvh8.cuh"

constexpr unsigned int all_threads = 0xFFFFFFFF;
constexpr unsigned long long invalid_task = ~0ull;

// Функция вычисляет масштаб для одной оси.
// Возвращает байт экспоненты и float-множитель для квантования.
__device__ __forceinline__ static void compute_quantization(float min_p, float max_p, uint8_t& exp_out, float& scale_out)
{
  float extent = max_p - min_p;
  if (extent <= 1e-8f) {
    exp_out = 127;
    scale_out = 1.0f;
    return;
  }

  float x = extent / 255.0f;
  int exp = static_cast<int>(ceilf(log2f(x)));
  exp_out = static_cast<uint8_t>(exp + 127);
  scale_out = exp2f(static_cast<float>(exp));
}

// Функции сжатия. Округляем ВНИЗ для минимума и ВВЕРХ для максимума (чтобы бокс не стал меньше)
__device__ __forceinline__ static uint8_t quantize_lo(float p, float p_base, float scale)
{
  float val = floorf((p - p_base) / scale);
  return static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
}

__device__ __forceinline__ static uint8_t quantize_hi(float p, float p_base, float scale)
{
  float val = ceilf((p - p_base) / scale);
  return static_cast<uint8_t>(fminf(fmaxf(val, 0.0f), 255.0f));
}

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

__device__ __forceinline__ static int find_largest_internal_frontier_node(
    const BVH2Node* bvh2_nodes,
    const unsigned int* frontier,
    unsigned int frontier_size)
{
  int best_index = -1;
  float best_area = -1.0f;
  for (unsigned int i = 0; i < frontier_size; ++i) {
    const BVH2Node& candidate = load_bvh2_node(bvh2_nodes, frontier[i]);
    if (candidate.is_leaf()) continue;
    const float area = candidate.aabb.surface_area();
    if (area > best_area) {
      best_area = area;
      best_index = static_cast<int>(i);
    }
  }
  return best_index;
}

namespace cuda::hploc
{
  __global__ void build_bvh8_kernel(
      const BVH2Node* __restrict__ bvh2_nodes,
      BVH8Node* __restrict__ bvh8_nodes,
      unsigned int* __restrict__ bvh8_prim_indices,
      volatile unsigned long long* __restrict__ tasks,
      unsigned int* __restrict__ n_tasks,
      unsigned int* __restrict__ n_bvh8_nodes,
      unsigned int* __restrict__ n_bvh8_leaves,
      unsigned int n_faces)
  {
    const unsigned int task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= n_faces) return;

    // Busy waiting тасок воркерами
    unsigned long long task = invalid_task;
    while (task == invalid_task) {
      task = tasks[task_id];
    }

    // Одну из тасок передаем сами себе для экономии
    while (true) {
      const unsigned int bvh2_node_index = unpack_bvh2_node_index(task);
      const unsigned int bvh8_node_index = unpack_bvh8_node_index(task);

      if (bvh8_node_index == INVALID_INDEX) {
        return;
      }

      const BVH2Node& binary_node = load_bvh2_node(bvh2_nodes, bvh2_node_index);

      // Если у нас больше одного примитива, то корень не может быть листом. Значит таску положил другой воркер.
      // Лист мы кладем только с bvh8_node_index = INVALID_INDEX, а этот случай уже был пройден выше.
      curassert(!binary_node.is_leaf(), 24341771);

      // Ищем своих детей: расширяем бинарную ноду, записываем индексы во frontier
      unsigned int frontier[8];
      frontier[0] = binary_node.left_child_index;
      frontier[1] = binary_node.right_child_index;
      unsigned int frontier_size = 2;

      while (frontier_size < 8) {
        const int expand_index = find_largest_internal_frontier_node(bvh2_nodes, frontier, frontier_size);
        if (expand_index < 0) break;

        BVH2Node expanded = load_bvh2_node(bvh2_nodes, frontier[expand_index]);
        frontier[expand_index] = expanded.left_child_index;
        frontier[frontier_size++] = expanded.right_child_index;
      }

      // 1. Инициализируем базовую точку родителя (p)
      BVH8Node bvh8_node{};
      bvh8_node.p_x = binary_node.aabb.min_x;
      bvh8_node.p_y = binary_node.aabb.min_y;
      bvh8_node.p_z = binary_node.aabb.min_z;

      // 2. Считаем масштаб и экспоненты
      float scale_x, scale_y, scale_z;
      compute_quantization(binary_node.aabb.min_x, binary_node.aabb.max_x, bvh8_node.e_x, scale_x);
      compute_quantization(binary_node.aabb.min_y, binary_node.aabb.max_y, bvh8_node.e_y, scale_y);
      compute_quantization(binary_node.aabb.min_z, binary_node.aabb.max_z, bvh8_node.e_z, scale_z);

      // 3. Считаем, сколько у нас внутренних узлов и листьев
      unsigned int n_internal = 0;
      unsigned int n_leaves = 0;
      for (unsigned int i = 0; i < frontier_size; ++i) {
        if (load_bvh2_node(bvh2_nodes, frontier[i]).is_leaf())
          n_leaves++;
        else
          n_internal++;
      }

      // 4. Выделяем память (базовые индексы) один раз на всю ноду
      bvh8_node.child_base_idx = atomicAdd(n_bvh8_nodes, n_internal);
      bvh8_node.prim_base_idx = (n_leaves > 0) ? atomicAdd(n_bvh8_leaves, n_leaves) : 0;
      bvh8_node.imask = 0;

      const unsigned int new_tasks_start = atomicAdd(n_tasks, frontier_size - 1u);
      unsigned int next_internal_offset = 0;
      unsigned int next_leaf_offset = 0;

      // 5. Заполняем слоты (прямая укладка, без хитрой сортировки)
      for (unsigned int i = 0; i < 8; ++i) {
        bvh8_node.meta[i] = 0;  // Инициализируем нулем

        if (i >= frontier_size) continue;  // Пустой слот

        const BVH2Node& child = load_bvh2_node(bvh2_nodes, frontier[i]);

        // Квантуем AABB ребенка
        bvh8_node.q_lo_x[i] = quantize_lo(child.aabb.min_x, bvh8_node.p_x, scale_x);
        bvh8_node.q_hi_x[i] = quantize_hi(child.aabb.max_x, bvh8_node.p_x, scale_x);
        bvh8_node.q_lo_y[i] = quantize_lo(child.aabb.min_y, bvh8_node.p_y, scale_y);
        bvh8_node.q_hi_y[i] = quantize_hi(child.aabb.max_y, bvh8_node.p_y, scale_y);
        bvh8_node.q_lo_z[i] = quantize_lo(child.aabb.min_z, bvh8_node.p_z, scale_z);
        bvh8_node.q_hi_z[i] = quantize_hi(child.aabb.max_z, bvh8_node.p_z, scale_z);

        unsigned long long new_task = invalid_task;

        if (child.is_leaf()) {
          uint8_t count = 0b001;
          bvh8_node.meta[i] = (count << 5) | (next_leaf_offset & 0x1F);
          bvh8_prim_indices[bvh8_node.prim_base_idx + next_leaf_offset] = child.right_child_index;
          next_leaf_offset++;
          new_task = pack_task(frontier[i], INVALID_INDEX);
        } else {
          bvh8_node.imask |= (1u << i);
          bvh8_node.meta[i] = (0b001 << 5) | ((24 + i) & 0x1F);
          unsigned int child_idx = bvh8_node.child_base_idx + next_internal_offset;
          next_internal_offset++;
          new_task = pack_task(frontier[i], child_idx);
        }

        // Раздаем таски
        if (i == 0)
          task = new_task;
        else
          tasks[new_tasks_start + i - 1] = new_task;
      }

      // 6. Сохраняем готовую 80-байтную ноду в память
      bvh8_nodes[bvh8_node_index] = bvh8_node;
    }
  }
}  // namespace cuda::hploc
