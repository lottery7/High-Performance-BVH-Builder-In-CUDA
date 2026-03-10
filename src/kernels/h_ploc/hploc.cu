#include <cooperative_groups/details/helpers.h>
#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../kernels.h"
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

__device__ __forceinline__ int lcp(const MortonCode* morton_codes, int n, int i, int j)
{
  if (i < 0 || j < 0 || i >= n || j >= n) return -1;
  if (morton_codes[i] == morton_codes[j]) return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
  return __clz(morton_codes[i] ^ morton_codes[j]);
}

__device__ __forceinline__ static unsigned int find_parent_id(
    const MortonCode* morton_codes,
    unsigned int n_faces,
    unsigned int left,
    unsigned int right)
{
  if (left == 0 || (right != n_faces - 1 && lcp(morton_codes, n_faces, right, right + 1) < lcp(morton_codes, n_faces, left - 1, left))) return right;
  return left - 1;
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
  // Количество валидных класетров во всем ворпе
  unsigned int n_valid_clusters = __popc(__ballot_sync(ALL_THREADS, is_valid && cluster_id != INVALID_INDEX));
  return n_valid_clusters;
}

__device__ __forceinline__ AABB shfl_aabb(unsigned mask, AABB aabb, int srcLane)
{
  return {
      __shfl_sync(mask, aabb.min_x, srcLane),
      __shfl_sync(mask, aabb.min_y, srcLane),
      __shfl_sync(mask, aabb.min_z, srcLane),
      __shfl_sync(mask, aabb.max_x, srcLane),
      __shfl_sync(mask, aabb.max_y, srcLane),
      __shfl_sync(mask, aabb.max_z, srcLane),
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
  bool is_active = lane_id < warp_n_clusters;
  // lane_id ближайшего соседа ближайшего соседа
  unsigned int nn_nn_lane_id = __shfl_sync(ALL_THREADS, nn_lane_id, nn_lane_id);
  // Проверка NN[NN[i]] == i
  bool is_mutual_nn = is_active && lane_id == nn_nn_lane_id;
  // Мержим только левым ребенком
  bool should_merge = is_mutual_nn && lane_id < nn_lane_id;

  unsigned int merge_mask = __ballot_sync(ALL_THREADS, should_merge);
  unsigned int merges_count = __popc(merge_mask);

  // Делаем atomicAdd одним потоком ворпа, а остальные читают из него
  unsigned int cluster_start_id = INVALID_INDEX;
  if (lane_id == 0) cluster_start_id = atomicAdd(n_clusters, merges_count);
  cluster_start_id = __shfl_sync(ALL_THREADS, cluster_start_id, 0);
  curassert(cluster_start_id != INVALID_INDEX, 66602491);

  // Число потоков ворпа с id меньше нашего, которые тоже будут делать merge
  unsigned int lower = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
  unsigned int relative_cluster_id = __popc(merge_mask & lower);
  // unsigned int relative_cluster_id = __popc(merge_mask << (WARP_SIZE - lane_id));
  unsigned int neighbor_cluster_id = __shfl_sync(ALL_THREADS, cluster_id, nn_lane_id);
  AABB neighbor_cluster_aabb = shfl_aabb(ALL_THREADS, cluster_aabb, nn_lane_id);

  if (should_merge) {
    cluster_aabb = AABB::union_of(cluster_aabb, neighbor_cluster_aabb);
    BVHNode node{cluster_aabb, cluster_id, neighbor_cluster_id};
    cluster_id = cluster_start_id + relative_cluster_id;
    nodes[cluster_id] = node;
  }

  // Для обеспечения occupancy убираем потоки, которые соответствуют смерженым кластерам (из двух оставляем один)
  unsigned int active_mask = __ballot_sync(ALL_THREADS, should_merge || !is_mutual_nn);

  // find nth set bit - n-ый включенный бит - lane_id потока, который должен занять наш поток
  unsigned int new_lane_id = __fns(active_mask, 0, lane_id + 1);
  cluster_id = __shfl_sync(ALL_THREADS, cluster_id, new_lane_id);
  if (new_lane_id == INVALID_INDEX) cluster_id = INVALID_INDEX;
  cluster_aabb = shfl_aabb(ALL_THREADS, cluster_aabb, new_lane_id);

  return warp_n_clusters - merges_count;
}

__device__ __forceinline__ static AABB shfl_down_sync(unsigned int mask, AABB value, unsigned int delta)
{
  float min_x = __shfl_down_sync(mask, value.min_x, delta);
  float min_y = __shfl_down_sync(mask, value.min_y, delta);
  float min_z = __shfl_down_sync(mask, value.min_z, delta);
  float max_x = __shfl_down_sync(mask, value.max_x, delta);
  float max_y = __shfl_down_sync(mask, value.max_y, delta);
  float max_z = __shfl_down_sync(mask, value.max_z, delta);
  return {min_x, min_y, min_z, max_x, max_y, max_z};
}

__device__ static unsigned int find_nearest_neighbor(unsigned int warp_n_clusters, AABB cluster_aabb)
{
  curassert(warp_n_clusters <= WARP_SIZE, 81387392);
  unsigned int lane_id = get_lane_id();
  unsigned int active_mask = __ballot_sync(ALL_THREADS, lane_id < warp_n_clusters);

  // TODO uint2 в оригинале - возможно быстрее за счет сравнения интов вместо флоатов
  float min_area = FLT_MAX;
  // lane_id соседа, объединение с которым дает min_area (nearest neighbor's lane id)
  unsigned int nn_lane_id = INVALID_INDEX;

  // #pragma unroll TODO
  for (unsigned int r = 1; r <= SEARCH_RADIUS; r++) {
    // === Поиск вправо (+ r)
    unsigned int neighbor_lane_id = lane_id + r;
    // Обновляем минимум у lane_id
    AABB neighbor_aabb = shfl_down_sync(active_mask, cluster_aabb, r);
    float union_area = FLT_MAX;  // Площадь поверхности объединения текущего и соседнего кластеров (lane_id U [lane_id + r])
    if (neighbor_lane_id < warp_n_clusters) {
      union_area = AABB::union_of(cluster_aabb, neighbor_aabb).surface_area();
      // Update min_distance[i, i + r]
      if (union_area < min_area) {
        min_area = union_area;
        nn_lane_id = neighbor_lane_id;
      }
    }

    // === Поиск влево (- r)
    // Обновляем минимум у lane_id + r (мы для него - lane_id - r)
    // Текущий минимум у lane_id + r
    float neighbor_min_area = __shfl_down_sync(active_mask, min_area, r);
    unsigned int neighbor_nn_lane_id = __shfl_down_sync(active_mask, nn_lane_id, r);
    if (neighbor_lane_id < warp_n_clusters && union_area < neighbor_min_area) {
      neighbor_min_area = union_area;
      neighbor_nn_lane_id = lane_id;
    }
    // Тут в lane_id + r мы получаем минимум из lane_id, по сути передаем обновленный минимум в lane_id + r
    float left_neighbor_min_area = __shfl_up_sync(active_mask, neighbor_min_area, r);
    unsigned int left_neighbor_nn_lane_id = __shfl_up_sync(active_mask, neighbor_nn_lane_id, r);
    if (lane_id < warp_n_clusters && lane_id >= r) {
      min_area = left_neighbor_min_area;
      nn_lane_id = left_neighbor_nn_lane_id;
    }
  }

  return nn_lane_id;
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
  // Получаем границы узла из thread_id
  unsigned int l_start = __shfl_sync(ALL_THREADS, left, target_lane_id);
  unsigned int l_end = __shfl_sync(ALL_THREADS, split, target_lane_id);
  unsigned int r_end = __shfl_sync(ALL_THREADS, right, target_lane_id) + 1;
  unsigned int r_start = l_end;

  unsigned int lane_id = get_lane_id();
  unsigned int cluster_id = INVALID_INDEX;

  curassert(l_start <= l_end, 150462);
  curassert(l_end <= r_end, 70862572);

  // Загружаем индексы кластеров из детей thread_id
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

  if (lane_id < warp_n_clusters) cluster_ids[l_start + lane_id] = cluster_id;
  __threadfence();  // для остальных ворпов
}

__global__ static void build_kernel(
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
      unsigned int lane_id = __ffs(warp_mask) - 1;

      if (get_lane_id() == lane_id) curassert(split != INVALID_INDEX, 94341937);
      ploc_merge(lane_id, left, right, split, is_final, nodes, cluster_ids, n_clusters);

      // Обнуляем крайний правый бит
      warp_mask = warp_mask & (warp_mask - 1);
    }
  }
}

__device__ __forceinline__ static float fminf3(float x, float y, float z) { return fminf(fminf(x, y), z); }

__device__ __forceinline__ static float fmaxf3(float x, float y, float z) { return fmaxf(fmaxf(x, y), z); }

__global__ static void build_leaves_nodes_kernel(const unsigned int* faces, unsigned int n_faces, const float* vertices, BVHNode* nodes)
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

namespace cuda::hploc
{
  void build(
      cudaStream_t stream,
      AABB scene_aabb,
      unsigned int* d_faces,
      float* d_vertices,
      BVHNode* d_nodes,
      unsigned int* d_parents,
      MortonCode* d_morton_codes,
      unsigned int* d_cluster_ids,
      unsigned int* d_n_clusters,
      unsigned int n_faces)
  {
    build_leaves_nodes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(d_faces, n_faces, d_vertices, d_nodes);
    fill_indices(stream, d_cluster_ids, n_faces);
    compute_morton_codes(stream, scene_aabb, d_faces, d_vertices, d_morton_codes, n_faces);
    sort_by_key(stream, d_morton_codes, d_cluster_ids, n_faces);

    cudaMemsetAsync(d_parents, INVALID_INDEX, sizeof(unsigned int) * (2 * n_faces - 1), stream);
    cudaMemcpyAsync(d_n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream);  // изначально n_faces кластеров - по одному на примитив
    constexpr size_t block_sz = 128;
    build_kernel<<<div_ceil(n_faces, block_sz), block_sz, 0, stream>>>(d_parents, d_morton_codes, d_nodes, d_cluster_ids, d_n_clusters, n_faces);
  }
}  // namespace cuda::hploc
