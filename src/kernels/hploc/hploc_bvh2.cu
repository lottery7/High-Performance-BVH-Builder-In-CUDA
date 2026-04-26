#include <cooperative_groups/details/helpers.h>
#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/helpers.cuh"
#include "../structs/aabb.h"
#include "../structs/morton_code.h"

#define SEARCH_RADIUS 8
#define WARP_SIZE 32
#define MERGING_THRESHOLD 16
#define ALL_THREADS 0xFFFFFFFFu

__device__ __forceinline__ static unsigned int get_lane_id()
{
  // threadIdx.x % WARP_SIZE
  // return threadIdx.x & (WARP_SIZE - 1);
  // TODO Эффект не заметен, но в теории должно помогать (меньше инструкций)
  unsigned int r;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(r));
  return r;
}

__device__ __forceinline__ static unsigned lanemask_lt()
{
  // TODO Эквивалент (1u << lane_id) - 1), помогло
  unsigned int r;
  asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(r));
  return r;
}

// Потоковая 128-битная запись (Обходит L1/L2 кэш, спасая его для vertices)
__device__ __forceinline__ void stream_store_uint4(void* dst, const uint4& src)
{
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(dst), "r"(src.x), "r"(src.y), "r"(src.z), "r"(src.w) : "memory");
}

// Потоковая 32-битная запись
__device__ __forceinline__ void stream_store_uint(unsigned int* dst, unsigned int src)
{
  asm volatile("st.global.cs.u32 [%0], %1;" : : "l"(dst), "r"(src) : "memory");
}

// Экстремально быстрая математика площади через FMA
__device__ __forceinline__ static float fused_half_area(const AABB& a, const AABB& b)
{
  float dx = fmaxf(a.max_x, b.max_x) - fminf(a.min_x, b.min_x);
  float dy = fmaxf(a.max_y, b.max_y) - fminf(a.min_y, b.min_y);
  float dz = fmaxf(a.max_z, b.max_z) - fminf(a.min_z, b.min_z);
  return fmaf(dz, dx + dy, dx * dy);
}

__device__ __forceinline__ static unsigned long long delta(const MortonCode* morton_codes, unsigned int l, unsigned int r)
{
  // TODO ldg кеш помог
  return (static_cast<unsigned long long>(__ldg(&morton_codes[l])) << 32 | l) ^ (static_cast<unsigned long long>(__ldg(&morton_codes[r])) << 32 | r);
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
    const unsigned int* __restrict__ cluster_ids,
    unsigned int& cluster_id)
{
  curassert(start <= end, 2349005);
  unsigned int index = get_lane_id() - offset;
  bool is_valid = index < min(end - start, MERGING_THRESHOLD);
  // TODO тут ldg кеш не изменил скорости работы
  if (is_valid) cluster_id = cluster_ids[start + index];
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

__device__ __forceinline__ static unsigned int merge_clusters_create_bvh2_node(
    unsigned int warp_n_clusters,
    unsigned int nn_lane_id,
    unsigned int& cluster_id,
    AABB& cluster_aabb,
    unsigned int* __restrict__ n_clusters,
    BVH2Node* __restrict__ nodes)
{
  unsigned int lane_id = get_lane_id();
  unsigned int nn_nn_lane_id = __shfl_sync(ALL_THREADS, nn_lane_id, nn_lane_id);
  bool is_mutual_nn = lane_id < warp_n_clusters && lane_id == nn_nn_lane_id;
  bool should_merge = is_mutual_nn && lane_id < nn_lane_id;

  unsigned int merge_mask = __ballot_sync(ALL_THREADS, should_merge);
  unsigned int merges_count = __popc(merge_mask);

  unsigned int cluster_start_id = INVALID_INDEX;
  // TODO merges_count > 0 сделало ХУЖЕ
  if (lane_id == 0) cluster_start_id = atomicAdd(n_clusters, merges_count);
  cluster_start_id = __shfl_sync(ALL_THREADS, cluster_start_id, 0);
  curassert(cluster_start_id != INVALID_INDEX, 66602491);

  unsigned int neighbor_cluster_id = __shfl_sync(ALL_THREADS, cluster_id, nn_lane_id);
  AABB neighbor_cluster_aabb = shfl_sync(ALL_THREADS, cluster_aabb, nn_lane_id);

  if (should_merge) {
    cluster_aabb = AABB::union_of(cluster_aabb, neighbor_cluster_aabb);
    BVH2Node node{cluster_aabb, cluster_id, neighbor_cluster_id};
    // cluster_id = cluster_start_id + __popc(merge_mask & (1u << lane_id) - 1);
    cluster_id = cluster_start_id + __popc(merge_mask & lanemask_lt());

    // TODO Сильно помогло
    reinterpret_cast<uint4*>(&nodes[cluster_id])[0] = reinterpret_cast<uint4*>(&node)[0];
    reinterpret_cast<uint4*>(&nodes[cluster_id])[1] = reinterpret_cast<uint4*>(&node)[1];
  }

  const unsigned int active_mask = __ballot_sync(ALL_THREADS, should_merge || !is_mutual_nn);

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
  // TODO Замена на unsigned long long не помогла
  uint2 nearest = make_uint2(INVALID_INDEX, INVALID_INDEX);

  // #pragma unroll TODO Не помогло
  for (unsigned int radius = 1; radius <= SEARCH_RADIUS; ++radius) {
    const unsigned int neighbor_index = lane_id + radius;
    unsigned int area = ~0u;
    AABB right_neighbor_aabb = shfl_sync(ALL_THREADS, cluster_aabb, neighbor_index);

    if (neighbor_index < warp_n_clusters) {
      area = __float_as_uint(fused_half_area(right_neighbor_aabb, cluster_aabb));
      if (area < nearest.x) nearest = make_uint2(area, neighbor_index);
    }

    uint2 neighbor_nn = shfl_sync(ALL_THREADS, nearest, neighbor_index);
    if (area < neighbor_nn.x) neighbor_nn = make_uint2(area, lane_id);

    nearest = shfl_sync(ALL_THREADS, neighbor_nn, lane_id - radius);
  }

  return nearest.y;
}

__device__ __forceinline__ static void ploc_merge(
    unsigned int target_lane_id,
    unsigned int left,
    unsigned int right,
    unsigned int split,
    bool is_final,
    BVH2Node* __restrict__ nodes,
    unsigned int* __restrict__ cluster_ids,
    unsigned int* __restrict__ n_clusters)
{
  unsigned int l_start = __shfl_sync(ALL_THREADS, left, target_lane_id);
  unsigned int l_end = __shfl_sync(ALL_THREADS, split, target_lane_id);
  unsigned int r_end = __shfl_sync(ALL_THREADS, right, target_lane_id) + 1;
  unsigned int r_start = l_end;
  unsigned int lane_id = get_lane_id();
  unsigned int cluster_id = INVALID_INDEX;

  curassert(l_start <= l_end, 150462);
  curassert(l_end <= r_end, 70862572);

  unsigned int n_left_clusters = load_cluster_id(l_start, l_end, 0, cluster_ids, cluster_id);
  unsigned int n_right_clusters = load_cluster_id(r_start, r_end, n_left_clusters, cluster_ids, cluster_id);
  unsigned int warp_n_clusters = n_left_clusters + n_right_clusters;

  AABB cluster_aabb{};  // TODO Фигурные скобки реально помогают
  if (lane_id < warp_n_clusters) {
    cluster_aabb = nodes[cluster_id].aabb;
  }

  unsigned int threshold = __shfl_sync(ALL_THREADS, is_final, target_lane_id) ? 1 : MERGING_THRESHOLD;

  while (warp_n_clusters > threshold) {
    unsigned int nn_lane_id = find_nearest_neighbor(warp_n_clusters, cluster_aabb);
    warp_n_clusters = merge_clusters_create_bvh2_node(warp_n_clusters, nn_lane_id, cluster_id, cluster_aabb, n_clusters, nodes);
  }

  if (lane_id < n_left_clusters + n_right_clusters) cluster_ids[l_start + lane_id] = cluster_id;
  __threadfence();
}

__device__ __forceinline__ static void atomic_min_float(float* ptr, float value)
{
  unsigned int current = atomicAdd(reinterpret_cast<unsigned int*>(ptr), 0);
  while (value < __int_as_float(current)) {
    const unsigned int previous = current;
    current = atomicCAS(reinterpret_cast<unsigned int*>(ptr), current, __float_as_int(value));
    if (current == previous) break;
  }
}

__device__ __forceinline__ static void atomic_max_float(float* ptr, float value)
{
  unsigned int current = atomicAdd(reinterpret_cast<unsigned int*>(ptr), 0);
  while (value > __int_as_float(current)) {
    const unsigned int previous = current;
    current = atomicCAS(reinterpret_cast<unsigned int*>(ptr), current, __float_as_int(value));
    if (current == previous) break;
  }
}

__device__ __forceinline__ static void atomic_grow(AABB* dst, const AABB& src)
{
  atomic_min_float(&dst->min_x, src.min_x);
  atomic_min_float(&dst->min_y, src.min_y);
  atomic_min_float(&dst->min_z, src.min_z);
  atomic_max_float(&dst->max_x, src.max_x);
  atomic_max_float(&dst->max_y, src.max_y);
  atomic_max_float(&dst->max_z, src.max_z);
}

__device__ __forceinline__ static AABB shfl_down_sync_aabb(unsigned int mask, AABB value, unsigned int delta)
{
  return {
      __shfl_down_sync(mask, value.min_x, delta),
      __shfl_down_sync(mask, value.min_y, delta),
      __shfl_down_sync(mask, value.min_z, delta),
      __shfl_down_sync(mask, value.max_x, delta),
      __shfl_down_sync(mask, value.max_y, delta),
      __shfl_down_sync(mask, value.max_z, delta),
  };
}

__device__ __forceinline__ static AABB warp_reduce_grow(AABB bounds)
{
  bounds = AABB::union_of(bounds, shfl_down_sync_aabb(ALL_THREADS, bounds, 16));
  bounds = AABB::union_of(bounds, shfl_down_sync_aabb(ALL_THREADS, bounds, 8));
  bounds = AABB::union_of(bounds, shfl_down_sync_aabb(ALL_THREADS, bounds, 4));
  bounds = AABB::union_of(bounds, shfl_down_sync_aabb(ALL_THREADS, bounds, 2));
  bounds = AABB::union_of(bounds, shfl_down_sync_aabb(ALL_THREADS, bounds, 1));
  return bounds;
}

namespace cuda::hploc
{
  __global__ void build_kernel(
      unsigned int* __restrict__ parents,
      const MortonCode* __restrict__ morton_codes,
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ cluster_ids,
      unsigned int* __restrict__ n_clusters,
      unsigned int n_faces)
  {
    const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int left = index;
    unsigned int right = index;
    unsigned int split = INVALID_INDEX;
    bool is_active = index < n_faces;

    while (__ballot_sync(ALL_THREADS, is_active)) {
      if (is_active) {
        unsigned int parent = find_parent_id(morton_codes, n_faces, left, right);
        bool is_right_child = (parent == right);

        // TODO Branchless вычисление аргументов для атомика - помогло
        unsigned int atomic_idx = is_right_child ? right : (left - 1);
        unsigned int atomic_val = is_right_child ? left : right;

        unsigned int prev_id = atomicExch(&parents[atomic_idx], atomic_val);

        if (prev_id != INVALID_INDEX) {
          split = is_right_child ? (right + 1) : left;
          left = is_right_child ? left : prev_id;
          right = is_right_child ? prev_id : right;
        } else {
          is_active = false;
        }
      }

      unsigned int size = right - left + 1;
      bool is_final = is_active && size == n_faces;
      unsigned int warp_mask = __ballot_sync(ALL_THREADS, is_active && (size > MERGING_THRESHOLD) || is_final);

      while (warp_mask) {
        unsigned int target_lane_id = __ffs(warp_mask) - 1;
        curassert(get_lane_id() != target_lane_id || split != INVALID_INDEX, 94341937);
        ploc_merge(target_lane_id, left, right, split, is_final, nodes, cluster_ids, n_clusters);
        warp_mask &= (warp_mask - 1);
      }
    }
  }

  __global__ void build_leaves_kernel(
      const unsigned int* __restrict__ faces,
      const unsigned int n_faces,
      const float* __restrict__ vertices,
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ clusters,
      AABB* __restrict__ scene_aabb)
  {
    const unsigned int base_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int thread_count = blockDim.x * gridDim.x;
    const unsigned int lane_id = get_lane_id();

    AABB local_bounds = AABB::neutral();

    // TODO GRID-STRIDE LOOP - кеширование + планировщик, помогает
    for (unsigned int index = base_index; index < n_faces; index += thread_count) {
      const unsigned int base_face = index * 3;
      const unsigned int f0 = __ldg(&faces[base_face + 0]);
      const unsigned int f1 = __ldg(&faces[base_face + 1]);
      const unsigned int f2 = __ldg(&faces[base_face + 2]);

      const float v0x = __ldg(&vertices[f0 * 3 + 0]);
      const float v1x = __ldg(&vertices[f1 * 3 + 0]);
      const float v2x = __ldg(&vertices[f2 * 3 + 0]);

      const float v0y = __ldg(&vertices[f0 * 3 + 1]);
      const float v1y = __ldg(&vertices[f1 * 3 + 1]);
      const float v2y = __ldg(&vertices[f2 * 3 + 1]);

      const float v0z = __ldg(&vertices[f0 * 3 + 2]);
      const float v1z = __ldg(&vertices[f1 * 3 + 2]);
      const float v2z = __ldg(&vertices[f2 * 3 + 2]);

      AABB aabb;
      aabb.min_x = fminf(fminf(v0x, v1x), v2x);
      aabb.min_y = fminf(fminf(v0y, v1y), v2y);
      aabb.min_z = fminf(fminf(v0z, v1z), v2z);
      aabb.max_x = fmaxf(fmaxf(v0x, v1x), v2x);
      aabb.max_y = fmaxf(fmaxf(v0y, v1y), v2y);
      aabb.max_z = fmaxf(fmaxf(v0z, v1z), v2z);

      local_bounds = AABB::union_of(local_bounds, aabb);

      // TODO Очень сильно ускоряет (4.7 ms -> 3.8 ms)
      BVH2Node node{aabb, INVALID_INDEX, index};
      reinterpret_cast<uint4*>(&nodes[index])[0] = reinterpret_cast<const uint4*>(&node)[0];
      reinterpret_cast<uint4*>(&nodes[index])[1] = reinterpret_cast<const uint4*>(&node)[1];
      clusters[index] = index;
    }

    local_bounds = warp_reduce_grow(local_bounds);
    if (lane_id == 0) {
      atomic_grow(scene_aabb, local_bounds);
    }
  }
}  // namespace cuda::hploc
