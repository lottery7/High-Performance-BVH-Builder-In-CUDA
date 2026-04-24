#include <cooperative_groups/details/helpers.h>
#include <cuda_runtime.h>
#include <float.h>

#include "../../utils/utils.h"
#include "../helpers/helpers.cuh"
#include "../structs/aabb.h"
#include "../structs/morton_code.h"

#define SEARCH_RADIUS 8
#define WARP_SIZE 32
#define ALL_THREADS 0xFFFFFFFFu

__device__ __forceinline__ static unsigned int get_lane_id()
{
  // threadIdx.x % WARP_SIZE
  return threadIdx.x & (WARP_SIZE - 1);
}

__device__ __forceinline__ static unsigned long long delta(const MortonCode* morton_codes, unsigned int l, unsigned int r)
{
  return (static_cast<unsigned long long>(morton_codes[l]) << 32 | l) ^ (static_cast<unsigned long long>(morton_codes[r]) << 32 | r);
}

__device__ __forceinline__ static unsigned int find_parent_id(const MortonCode* morton_codes, unsigned int n_faces, unsigned int l, unsigned int r)
{
  if (l == 0 || (r != n_faces - 1 && delta(morton_codes, r, r + 1) < delta(morton_codes, l - 1, l))) {
    return r;
  }
  return l - 1;
}

__device__ __forceinline__ static unsigned int load_cluster_id(
    unsigned int start,
    unsigned int end,
    int offset,
    const unsigned int* cluster_ids,
    unsigned int& cluster_id)
{
  curassert(start <= end, 2349005);
  int index = static_cast<int>(get_lane_id()) - offset;
  // Загружаем до WARP_SIZE / 2 кластеров
  bool is_valid = index >= 0 && index < min(end - start, WARP_SIZE / 2);
  if (is_valid) cluster_id = cluster_ids[start + index];
  // Количество валидных кластеров во всем ворпе
  unsigned int n_valid_clusters = __popc(__ballot_sync(ALL_THREADS, is_valid && cluster_id != INVALID_INDEX));
  return n_valid_clusters;
}

__device__ __forceinline__ static AABB shfl_sync(unsigned mask, AABB aabb, int src_lane)
{
  return {
      __shfl_sync(mask, aabb.min_x, src_lane),
      __shfl_sync(mask, aabb.min_y, src_lane),
      __shfl_sync(mask, aabb.min_z, src_lane),
      __shfl_sync(mask, aabb.max_x, src_lane),
      __shfl_sync(mask, aabb.max_y, src_lane),
      __shfl_sync(mask, aabb.max_z, src_lane),
  };
}

__device__ static unsigned int merge_clusters_create_bvh2_node(
    unsigned int warp_n_clusters,
    unsigned int nn_lane_id,  // nearest neighbor's lane id
    unsigned int& cluster_id,
    AABB& cluster_aabb,
    unsigned int* n_clusters,
    BVHNode* nodes)
{
  unsigned int lane_id = get_lane_id();
  // lane_id ближайшего соседа ближайшего соседа
  unsigned int nn_nn_lane_id = __shfl_sync(ALL_THREADS, nn_lane_id, nn_lane_id);
  // Проверка NN[NN[i]] == i
  bool is_mutual_nn = lane_id < warp_n_clusters && lane_id == nn_nn_lane_id;
  // Мержим только левым ребенком
  bool should_merge = is_mutual_nn && lane_id < nn_lane_id;

  unsigned int merge_mask = __ballot_sync(ALL_THREADS, should_merge);
  unsigned int merges_count = __popc(merge_mask);

  unsigned int cluster_start_id = INVALID_INDEX;
  if (lane_id == 0) cluster_start_id = atomicAdd(n_clusters, merges_count);
  cluster_start_id = __shfl_sync(ALL_THREADS, cluster_start_id, 0);
  curassert(cluster_start_id != INVALID_INDEX, 66602491);

  unsigned int neighbor_cluster_id = __shfl_sync(ALL_THREADS, cluster_id, nn_lane_id);
  AABB neighbor_cluster_aabb = shfl_sync(ALL_THREADS, cluster_aabb, nn_lane_id);

  if (should_merge) {
    cluster_aabb = AABB::union_of(cluster_aabb, neighbor_cluster_aabb);
    BVHNode node{cluster_aabb, cluster_id, neighbor_cluster_id};
    // Число потоков ворпа с id меньше нашего, которые тоже будут делать merge
    cluster_id = cluster_start_id + __popc(merge_mask & (1u << lane_id) - 1);
    nodes[cluster_id] = node;
  }

  // Для обеспечения occupancy убираем потоки, которые соответствуют смерженым кластерам (из двух оставляем один)
  const unsigned int active_mask = __ballot_sync(ALL_THREADS, should_merge || !is_mutual_nn);

  // find nth set bit - n-ый включенный бит - lane_id потока, который должен занять наш поток
  const int new_lane_id = __fns(active_mask, 0, lane_id + 1);
  cluster_id = __shfl_sync(ALL_THREADS, cluster_id, new_lane_id);
  if (new_lane_id == -1) cluster_id = INVALID_INDEX;
  cluster_aabb = shfl_sync(ALL_THREADS, cluster_aabb, new_lane_id);

  return warp_n_clusters - merges_count;
}

__device__ __forceinline__ unsigned long long float_as_uint64(float f) { return __float_as_uint(f); }

__device__ __forceinline__ uint2 shfl_sync(unsigned int mask, uint2 value, unsigned int src_lane)
{
  return make_uint2(__shfl_sync(mask, value.x, static_cast<int>(src_lane)), __shfl_sync(mask, value.y, static_cast<int>(src_lane)));
}

__device__ __forceinline__ unsigned int find_nearest_neighbor(unsigned int warp_n_clusters, AABB cluster_aabb)
{
  curassert(warp_n_clusters <= WARP_SIZE, 81387392);
  const unsigned int lane_id = get_lane_id();
  uint2 nearest = make_uint2(INVALID_INDEX, INVALID_INDEX);

  for (unsigned int radius = 1; radius <= SEARCH_RADIUS; ++radius) {
    const unsigned int neighbor_index = lane_id + radius;
    unsigned int area = ~0u;
    AABB right_neighbor_aabb = shfl_sync(ALL_THREADS, cluster_aabb, neighbor_index);

    if (neighbor_index < warp_n_clusters) {
      area = __float_as_uint(AABB::union_of(right_neighbor_aabb, cluster_aabb).surface_area());
      if (area < nearest.x) nearest = make_uint2(area, neighbor_index);
    }

    uint2 neighbor_nn = shfl_sync(ALL_THREADS, nearest, neighbor_index);
    if (area < neighbor_nn.x) neighbor_nn = make_uint2(area, lane_id);

    nearest = shfl_sync(ALL_THREADS, neighbor_nn, lane_id - radius);
  }

  return nearest.y;
}

__device__ static void ploc_merge(
    unsigned int target_lane_id,
    unsigned int left,
    unsigned int right,
    unsigned int split,
    bool is_final,
    BVHNode* nodes,
    unsigned int* cluster_ids,
    unsigned int* n_clusters)
{
  // Получаем границы узла из target_lane_id
  unsigned int l_start = __shfl_sync(ALL_THREADS, left, target_lane_id);
  unsigned int l_end = __shfl_sync(ALL_THREADS, split, target_lane_id);
  unsigned int r_end = __shfl_sync(ALL_THREADS, right, target_lane_id) + 1;
  unsigned int r_start = l_end;

  unsigned int lane_id = get_lane_id();
  unsigned int cluster_id = INVALID_INDEX;

  curassert(l_start <= l_end, 150462);
  curassert(l_end <= r_end, 70862572);

  // Загружаем индексы кластеров из детей target_lane_id
  unsigned int n_left_clusters = load_cluster_id(l_start, l_end, 0, cluster_ids, cluster_id);
  unsigned int n_right_clusters = load_cluster_id(r_start, r_end, n_left_clusters, cluster_ids, cluster_id);
  unsigned int warp_n_clusters = n_left_clusters + n_right_clusters;

  AABB cluster_aabb;
  if (lane_id < warp_n_clusters) {
    cluster_aabb = nodes[cluster_id].aabb;
  }

  // В корне мы хотим получить 1 кластер
  unsigned int threshold = __shfl_sync(ALL_THREADS, is_final, target_lane_id) ? 1 : WARP_SIZE / 2;

  while (warp_n_clusters > threshold) {
    unsigned int nn_lane_id = find_nearest_neighbor(warp_n_clusters, cluster_aabb);
    warp_n_clusters = merge_clusters_create_bvh2_node(warp_n_clusters, nn_lane_id, cluster_id, cluster_aabb, n_clusters, nodes);
  }

  if (lane_id < n_left_clusters + n_right_clusters) cluster_ids[l_start + lane_id] = cluster_id;
  __threadfence();  // для остальных ворпов
}

__device__ __forceinline__ static float fminf3(float x, float y, float z) { return fminf(fminf(x, y), z); }

__device__ __forceinline__ static float fmaxf3(float x, float y, float z) { return fmaxf(fmaxf(x, y), z); }

namespace cuda::hploc
{
  __global__ void build_kernel(
      unsigned int* parents,
      const MortonCode* morton_codes,
      BVHNode* nodes,
      unsigned int* cluster_ids,
      unsigned int* n_clusters,
      unsigned int n_faces)
  {
    const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Отрезок и сплит из LBVH
    unsigned int left = index;
    unsigned int right = index;

    // Инвариант: split принадлежит правому ребенок (а не левому, как в обычном LBVH)
    unsigned int split = INVALID_INDEX;
    bool is_active = index < n_faces;

    // ballot возвращает 32 бита - boolean с каждого потока в ворпе
    while (__ballot_sync(ALL_THREADS, is_active)) {
      if (is_active) {
        unsigned int prev_id;

        if (find_parent_id(morton_codes, n_faces, left, right) == right) {
          // родитель справа => текущий узел - левый ребенок
          prev_id = atomicExch(&parents[right], left);

          // Если что-то лежит, значит правый ребенок уже дошел и положил туда свою границу
          if (prev_id != INVALID_INDEX) {
            split = right + 1;
            right = prev_id;  // расширили текущий отрезок, поэтому split вычисляем ДО этой строки
          }
        } else {
          prev_id = atomicExch(&parents[left - 1], right);
          if (prev_id != INVALID_INDEX) {
            split = left;
            left = prev_id;
          }
        }

        // Если мы первые - отрубаемся
        if (prev_id == INVALID_INDEX) is_active = false;
      }

      curassert(left <= right, 67224631);
      unsigned int size = right - left + 1;

      bool is_final = is_active && size == n_faces;

      // Делаем слияние кластеров в двух случаях:
      // 1. У текущего узла слишком большое количество примитивов
      // 2. Мы дошли до корня и надо сделать финальное слияние
      unsigned int warp_mask = __ballot_sync(ALL_THREADS, is_active && (size > WARP_SIZE / 2) || is_final);
      while (warp_mask) {
        // find first set bit - номер первого ненулевого бита, с индексацией с 1
        unsigned int target_lane_id = __ffs(warp_mask) - 1;

        curassert(get_lane_id() != target_lane_id || split != INVALID_INDEX, 94341937);
        ploc_merge(target_lane_id, left, right, split, is_final, nodes, cluster_ids, n_clusters);

        // Обнуляем крайний правый бит
        warp_mask = warp_mask & (warp_mask - 1);
      }
    }
  }
  __global__ void build_leaves_nodes_kernel(const unsigned int* faces, unsigned int n_faces, const float* vertices, BVHNode* nodes)
  {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;

    unsigned int f0 = faces[3 * index + 0];
    unsigned int f1 = faces[3 * index + 1];
    unsigned int f2 = faces[3 * index + 2];

    float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
    float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
    float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

    AABB aabb{
        fminf3(v0.x, v1.x, v2.x),
        fminf3(v0.y, v1.y, v2.y),
        fminf3(v0.z, v1.z, v2.z),
        fmaxf3(v0.x, v1.x, v2.x),
        fmaxf3(v0.y, v1.y, v2.y),
        fmaxf3(v0.z, v1.z, v2.z)};

    BVHNode node{aabb, INVALID_INDEX, index};
    nodes[index] = node;
  }
}  // namespace cuda::hploc
