#include "nexus_hploc.h"

#include <cub/device/device_radix_sort.cuh>

#include <cfloat>
#include <cstdint>

#include "../../utils/defines.h"
#include "../../utils/utils.h"

namespace
{
  constexpr unsigned int FULL_MASK = 0xffffffffu;
  constexpr unsigned int WARP_SIZE = 32u;
  constexpr unsigned int SEARCH_RADIUS = 8u;
  constexpr unsigned int MERGING_THRESHOLD = 16u;
  constexpr float SCENE_EPSILON = 1e-9f;

  struct BuildState {
    AABB* scene_bounds;
    BVHNode* nodes;
    unsigned int* cluster_indices;
    unsigned int* parent_indices;
    unsigned int prim_count;
    unsigned int* cluster_count;
  };

  __device__ __forceinline__ unsigned int get_lane_id() { return threadIdx.x & (WARP_SIZE - 1u); }

  __device__ __forceinline__ AABB make_empty_aabb()
  {
    return {FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  }

  __device__ __forceinline__ void grow_aabb(AABB& dst, const AABB& src) { dst = AABB::union_of(dst, src); }

  __device__ __forceinline__ float3 centroid_of(const AABB& aabb)
  {
    return make_float3(
        0.5f * (aabb.min_x + aabb.max_x),
        0.5f * (aabb.min_y + aabb.max_y),
        0.5f * (aabb.min_z + aabb.max_z));
  }

  __device__ __forceinline__ unsigned int ordered_positive_float_bits(float value) { return __float_as_uint(value); }

  __device__ __forceinline__ void atomic_min_float(float* ptr, float value)
  {
    unsigned int current = atomicAdd(reinterpret_cast<unsigned int*>(ptr), 0);
    while (value < __int_as_float(current)) {
      const unsigned int previous = current;
      current = atomicCAS(reinterpret_cast<unsigned int*>(ptr), current, __float_as_int(value));
      if (current == previous) break;
    }
  }

  __device__ __forceinline__ void atomic_max_float(float* ptr, float value)
  {
    unsigned int current = atomicAdd(reinterpret_cast<unsigned int*>(ptr), 0);
    while (value > __int_as_float(current)) {
      const unsigned int previous = current;
      current = atomicCAS(reinterpret_cast<unsigned int*>(ptr), current, __float_as_int(value));
      if (current == previous) break;
    }
  }

  __device__ __forceinline__ void atomic_grow(AABB* dst, const AABB& src)
  {
    atomic_min_float(&dst->min_x, src.min_x);
    atomic_min_float(&dst->min_y, src.min_y);
    atomic_min_float(&dst->min_z, src.min_z);
    atomic_max_float(&dst->max_x, src.max_x);
    atomic_max_float(&dst->max_y, src.max_y);
    atomic_max_float(&dst->max_z, src.max_z);
  }

  __device__ __forceinline__ AABB shfl_sync(unsigned int mask, AABB value, unsigned int src_lane)
  {
    return {
        __shfl_sync(mask, value.min_x, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.min_y, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.min_z, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.max_x, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.max_y, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.max_z, static_cast<int>(src_lane)),
    };
  }

  __device__ __forceinline__ AABB shfl_down_sync(unsigned int mask, AABB value, unsigned int delta)
  {
    return {
        __shfl_down_sync(mask, value.min_x, static_cast<int>(delta)),
        __shfl_down_sync(mask, value.min_y, static_cast<int>(delta)),
        __shfl_down_sync(mask, value.min_z, static_cast<int>(delta)),
        __shfl_down_sync(mask, value.max_x, static_cast<int>(delta)),
        __shfl_down_sync(mask, value.max_y, static_cast<int>(delta)),
        __shfl_down_sync(mask, value.max_z, static_cast<int>(delta)),
    };
  }

  __device__ __forceinline__ uint2 shfl_sync(unsigned int mask, uint2 value, unsigned int src_lane)
  {
    return make_uint2(
        __shfl_sync(mask, value.x, static_cast<int>(src_lane)),
        __shfl_sync(mask, value.y, static_cast<int>(src_lane)));
  }

  __device__ __forceinline__ AABB warp_reduce_grow(AABB bounds)
  {
    bounds = AABB::union_of(bounds, shfl_down_sync(FULL_MASK, bounds, 16));
    bounds = AABB::union_of(bounds, shfl_down_sync(FULL_MASK, bounds, 8));
    bounds = AABB::union_of(bounds, shfl_down_sync(FULL_MASK, bounds, 4));
    bounds = AABB::union_of(bounds, shfl_down_sync(FULL_MASK, bounds, 2));
    bounds = AABB::union_of(bounds, shfl_down_sync(FULL_MASK, bounds, 1));
    return bounds;
  }

  __device__ __forceinline__ float3 load_vertex(const float* vertices, unsigned int vertex_index)
  {
    return make_float3(vertices[3 * vertex_index + 0], vertices[3 * vertex_index + 1], vertices[3 * vertex_index + 2]);
  }

  __device__ __forceinline__ AABB make_triangle_aabb(const float* vertices, const unsigned int* faces, unsigned int face_index)
  {
    const unsigned int f0 = faces[3 * face_index + 0];
    const unsigned int f1 = faces[3 * face_index + 1];
    const unsigned int f2 = faces[3 * face_index + 2];

    const float3 v0 = load_vertex(vertices, f0);
    const float3 v1 = load_vertex(vertices, f1);
    const float3 v2 = load_vertex(vertices, f2);

    return {
        fminf(fminf(v0.x, v1.x), v2.x),
        fminf(fminf(v0.y, v1.y), v2.y),
        fminf(fminf(v0.z, v1.z), v2.z),
        fmaxf(fmaxf(v0.x, v1.x), v2.x),
        fmaxf(fmaxf(v0.y, v1.y), v2.y),
        fmaxf(fmaxf(v0.z, v1.z), v2.z),
    };
  }

  __device__ __forceinline__ unsigned int interleave_bits32(unsigned int x)
  {
    x = (x | (x << 16)) & 0x30000ffu;
    x = (x | (x << 8)) & 0x300f00fu;
    x = (x | (x << 4)) & 0x30c30c3u;
    x = (x | (x << 2)) & 0x9249249u;
    return x;
  }

  __device__ __forceinline__ unsigned int morton_code32(unsigned int x, unsigned int y, unsigned int z)
  {
    return interleave_bits32(x) | (interleave_bits32(y) << 1u) | (interleave_bits32(z) << 2u);
  }

  __device__ __forceinline__ unsigned int morton_code(const float3& centroid)
  {
    const unsigned int x = static_cast<unsigned int>(centroid.x * 0x3ffu);
    const unsigned int y = static_cast<unsigned int>(centroid.y * 0x3ffu);
    const unsigned int z = static_cast<unsigned int>(centroid.z * 0x3ffu);
    return morton_code32(x, y, z);
  }

  __global__ void compute_scene_bounds_kernel(BuildState build_state, const unsigned int* faces, const float* vertices)
  {
    const unsigned int prim_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int lane_id = get_lane_id();
    const unsigned int thread_count = blockDim.x * gridDim.x;

    AABB bounds = make_empty_aabb();

    for (unsigned int i = prim_index; i < build_state.prim_count; i += thread_count) {
      const AABB leaf_aabb = make_triangle_aabb(vertices, faces, i);
      build_state.nodes[i] = {leaf_aabb, INVALID_INDEX, i};
      grow_aabb(bounds, leaf_aabb);
    }

    bounds = warp_reduce_grow(bounds);
    if (lane_id == 0) atomic_grow(build_state.scene_bounds, bounds);
  }

  __global__ void compute_morton_codes_kernel(BuildState build_state, unsigned int* morton_codes)
  {
    const unsigned int prim_index = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int thread_count = blockDim.x * gridDim.x;

    const AABB scene_bounds = *build_state.scene_bounds;
    const float dx = fmaxf(scene_bounds.max_x - scene_bounds.min_x, SCENE_EPSILON);
    const float dy = fmaxf(scene_bounds.max_y - scene_bounds.min_y, SCENE_EPSILON);
    const float dz = fmaxf(scene_bounds.max_z - scene_bounds.min_z, SCENE_EPSILON);

    for (unsigned int i = prim_index; i < build_state.prim_count; i += thread_count) {
      const float3 centroid = centroid_of(build_state.nodes[i].aabb);
      const float3 normalized = make_float3(
          fminf(fmaxf((centroid.x - scene_bounds.min_x) / dx, 0.0f), 1.0f),
          fminf(fmaxf((centroid.y - scene_bounds.min_y) / dy, 0.0f), 1.0f),
          fminf(fmaxf((centroid.z - scene_bounds.min_z) / dz, 0.0f), 1.0f));

      morton_codes[i] = morton_code(normalized);
      build_state.cluster_indices[i] = i;
    }
  }

  __device__ __forceinline__ std::uint64_t delta(unsigned int a, unsigned int b, const unsigned int* morton_codes)
  {
    return (static_cast<std::uint64_t>(morton_codes[a]) << 32u | a) ^ (static_cast<std::uint64_t>(morton_codes[b]) << 32u | b);
  }

  __device__ __forceinline__ unsigned int find_parent_id(unsigned int left, unsigned int right, unsigned int prim_count, const unsigned int* morton_codes)
  {
    if (left == 0u || (right != prim_count - 1u && delta(right, right + 1u, morton_codes) < delta(left - 1u, left, morton_codes))) {
      return right;
    }
    return left - 1u;
  }

  __device__ __forceinline__ unsigned int load_indices(
      unsigned int start,
      unsigned int end,
      unsigned int& cluster_index,
      const BuildState& build_state,
      unsigned int offset)
  {
    const unsigned int lane_id = get_lane_id();
    const unsigned int index = lane_id - offset;
    const bool valid_lane = index < min(end - start, MERGING_THRESHOLD);

    if (valid_lane) cluster_index = build_state.cluster_indices[start + index];

    return __popc(__ballot_sync(FULL_MASK, valid_lane && cluster_index != INVALID_INDEX));
  }

  __device__ __forceinline__ void store_indices(unsigned int previous_num_prims, unsigned int cluster_index, const BuildState& build_state, unsigned int start)
  {
    const unsigned int lane_id = get_lane_id();
    if (lane_id < previous_num_prims) build_state.cluster_indices[start + lane_id] = cluster_index;
    __threadfence();
  }

  __device__ __forceinline__ unsigned int merge_clusters_create_bvh2_node(
      unsigned int num_prims,
      unsigned int nearest_neighbor,
      unsigned int& cluster_index,
      AABB& cluster_bounds,
      const BuildState& build_state)
  {
    const unsigned int lane_id = get_lane_id();
    const bool lane_active = lane_id < num_prims;

    const unsigned int nearest_neighbor_nn = __shfl_sync(FULL_MASK, nearest_neighbor, static_cast<int>(nearest_neighbor));
    const bool mutual_neighbor = lane_active && lane_id == nearest_neighbor_nn;
    const bool should_merge = mutual_neighbor && lane_id < nearest_neighbor;

    const unsigned int merge_mask = __ballot_sync(FULL_MASK, should_merge);
    const unsigned int merge_count = __popc(merge_mask);

    unsigned int base_index = 0;
    if (lane_id == 0u) base_index = atomicAdd(build_state.cluster_count, merge_count);
    base_index = __shfl_sync(FULL_MASK, base_index, 0);

    const unsigned int relative_index = __popc(merge_mask << (WARP_SIZE - lane_id));
    const unsigned int neighbor_cluster_index = __shfl_sync(FULL_MASK, cluster_index, static_cast<int>(nearest_neighbor));
    const AABB neighbor_bounds = shfl_sync(FULL_MASK, cluster_bounds, nearest_neighbor);

    if (should_merge) {
      cluster_bounds = AABB::union_of(cluster_bounds, neighbor_bounds);
      build_state.nodes[base_index + relative_index] = {cluster_bounds, cluster_index, neighbor_cluster_index};
      cluster_index = base_index + relative_index;
    }

    const unsigned int valid_mask = __ballot_sync(FULL_MASK, should_merge || !mutual_neighbor);
    const int shift = __fns(valid_mask, 0, static_cast<int>(lane_id) + 1);

    cluster_index = __shfl_sync(FULL_MASK, cluster_index, shift);
    if (shift == -1) cluster_index = INVALID_INDEX;
    cluster_bounds = shfl_sync(FULL_MASK, cluster_bounds, shift);

    return num_prims - merge_count;
  }

  __device__ __forceinline__ unsigned int find_nearest_neighbor(unsigned int num_prims, unsigned int cluster_index, AABB cluster_bounds, const BuildState& build_state)
  {
    (void)cluster_index;
    (void)build_state;

    const unsigned int lane_id = get_lane_id();
    uint2 min_area_index = make_uint2(INVALID_INDEX, INVALID_INDEX);

    for (unsigned int radius = 1; radius <= SEARCH_RADIUS; ++radius) {
      const unsigned int neighbor_index = lane_id + radius;
      unsigned int area = static_cast<unsigned int>(-1);
      AABB neighbor_bounds = shfl_sync(FULL_MASK, cluster_bounds, neighbor_index);

      if (neighbor_index < num_prims) {
        neighbor_bounds = AABB::union_of(neighbor_bounds, cluster_bounds);
        area = ordered_positive_float_bits(neighbor_bounds.surface_area());
        if (area < min_area_index.x) min_area_index = make_uint2(area, neighbor_index);
      }

      uint2 neighbor_nn = shfl_sync(FULL_MASK, min_area_index, neighbor_index);
      if (area < neighbor_nn.x) neighbor_nn = make_uint2(area, lane_id);

      min_area_index = shfl_sync(FULL_MASK, neighbor_nn, lane_id - radius);
    }

    return min_area_index.y;
  }

  __device__ __forceinline__ void ploc_merge(
      unsigned int lane_id,
      unsigned int left,
      unsigned int right,
      unsigned int split,
      bool final_merge,
      const BuildState& build_state)
  {
    const unsigned int left_start = __shfl_sync(FULL_MASK, left, static_cast<int>(lane_id));
    const unsigned int right_end = __shfl_sync(FULL_MASK, right, static_cast<int>(lane_id)) + 1u;
    const unsigned int left_end = __shfl_sync(FULL_MASK, split, static_cast<int>(lane_id));
    const unsigned int right_start = left_end;

    unsigned int cluster_index = INVALID_INDEX;
    const unsigned int num_left = load_indices(left_start, left_end, cluster_index, build_state, 0u);
    const unsigned int num_right = load_indices(right_start, right_end, cluster_index, build_state, num_left);
    unsigned int num_prims = num_left + num_right;

    AABB cluster_bounds = make_empty_aabb();
    if (get_lane_id() < num_prims) cluster_bounds = build_state.nodes[cluster_index].aabb;

    const unsigned int threshold = __shfl_sync(FULL_MASK, final_merge, static_cast<int>(lane_id)) ? 1u : MERGING_THRESHOLD;
    while (num_prims > threshold) {
      const unsigned int nearest_neighbor = find_nearest_neighbor(num_prims, cluster_index, cluster_bounds, build_state);
      num_prims = merge_clusters_create_bvh2_node(num_prims, nearest_neighbor, cluster_index, cluster_bounds, build_state);
    }

    store_indices(num_left + num_right, cluster_index, build_state, left_start);
  }

  __global__ void build_bvh2_kernel(BuildState build_state, const unsigned int* morton_codes)
  {
    const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int left = index;
    unsigned int right = index;
    unsigned int split = 0u;
    bool lane_active = index < build_state.prim_count;

    while (__ballot_sync(FULL_MASK, lane_active)) {
      if (lane_active) {
        unsigned int previous_id = INVALID_INDEX;

        if (find_parent_id(left, right, build_state.prim_count, morton_codes) == right) {
          previous_id = atomicExch(&build_state.parent_indices[right], left);
          if (previous_id != INVALID_INDEX) {
            split = right + 1u;
            right = previous_id;
          }
        } else {
          previous_id = atomicExch(&build_state.parent_indices[left - 1u], right);
          if (previous_id != INVALID_INDEX) {
            split = left;
            left = previous_id;
          }
        }

        if (previous_id == INVALID_INDEX) lane_active = false;
      }

      const unsigned int size = right - left + 1u;
      const bool final_merge = lane_active && size == build_state.prim_count;
      unsigned int warp_mask = __ballot_sync(FULL_MASK, (lane_active && size > MERGING_THRESHOLD) || final_merge);

      while (warp_mask != 0u) {
        const unsigned int lane_id = __ffs(warp_mask) - 1u;
        ploc_merge(lane_id, left, right, split, final_merge, build_state);
        warp_mask &= warp_mask - 1u;
      }
    }
  }
}  // namespace

namespace cuda::nexus_hploc
{
  namespace
  {
    struct EventPair {
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;

      EventPair()
      {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));
      }

      ~EventPair()
      {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
      }
    };

    double elapsed_ms(const EventPair& pair)
    {
      float elapsed_ms = 0.0f;
      CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_ms, pair.start, pair.stop));
      return static_cast<double>(elapsed_ms);
    }

    void build_impl(
        cudaStream_t stream,
        unsigned int* d_faces,
        float* d_vertices,
        BVHNode* d_nodes,
        Workspace& workspace,
        unsigned int n_faces,
        BuildTimings* timings)
    {
      BuildState build_state{};
      build_state.scene_bounds = workspace.d_scene_bounds;
      build_state.nodes = d_nodes;
      build_state.cluster_indices = workspace.d_cluster_indices;
      build_state.parent_indices = workspace.d_parent_indices;
      build_state.prim_count = n_faces;
      build_state.cluster_count = workspace.d_cluster_count;

      const AABB empty_scene_bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
      CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.scene_bounds, &empty_scene_bounds, sizeof(AABB), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemsetAsync(build_state.parent_indices, 0xff, sizeof(unsigned int) * n_faces, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.cluster_count, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

      if (timings == nullptr) {
        compute_scene_bounds_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, d_faces, d_vertices);
        compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, workspace.d_morton_codes);

        cub::DeviceRadixSort::SortPairs(
            workspace.d_sort_temp_storage,
            workspace.sort_temp_storage_bytes,
            workspace.d_morton_codes,
            workspace.d_morton_codes_sorted,
            workspace.d_cluster_indices,
            workspace.d_cluster_indices_sorted,
            n_faces,
            0,
            32,
            stream);

        build_state.cluster_indices = workspace.d_cluster_indices_sorted;
        constexpr unsigned int block_size = 64;
        build_bvh2_kernel<<<div_ceil(static_cast<int>(n_faces), static_cast<int>(block_size)), block_size, 0, stream>>>(
            build_state,
            workspace.d_morton_codes_sorted);
        return;
      }

      EventPair total;
      EventPair scene_bounds;
      EventPair morton;
      EventPair sort;
      EventPair build;

      CUDA_SAFE_CALL(cudaEventRecord(total.start, stream));

      CUDA_SAFE_CALL(cudaEventRecord(scene_bounds.start, stream));
      compute_scene_bounds_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, d_faces, d_vertices);
      CUDA_SAFE_CALL(cudaEventRecord(scene_bounds.stop, stream));

      CUDA_SAFE_CALL(cudaEventRecord(morton.start, stream));
      compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, workspace.d_morton_codes);
      CUDA_SAFE_CALL(cudaEventRecord(morton.stop, stream));

      CUDA_SAFE_CALL(cudaEventRecord(sort.start, stream));
      cub::DeviceRadixSort::SortPairs(
          workspace.d_sort_temp_storage,
          workspace.sort_temp_storage_bytes,
          workspace.d_morton_codes,
          workspace.d_morton_codes_sorted,
          workspace.d_cluster_indices,
          workspace.d_cluster_indices_sorted,
          n_faces,
          0,
          32,
          stream);
      CUDA_SAFE_CALL(cudaEventRecord(sort.stop, stream));

      build_state.cluster_indices = workspace.d_cluster_indices_sorted;
      constexpr unsigned int block_size = 64;
      CUDA_SAFE_CALL(cudaEventRecord(build.start, stream));
      build_bvh2_kernel<<<div_ceil(static_cast<int>(n_faces), static_cast<int>(block_size)), block_size, 0, stream>>>(
          build_state,
          workspace.d_morton_codes_sorted);
      CUDA_SAFE_CALL(cudaEventRecord(build.stop, stream));

      CUDA_SAFE_CALL(cudaEventRecord(total.stop, stream));

      CUDA_SAFE_CALL(cudaEventSynchronize(scene_bounds.stop));
      CUDA_SAFE_CALL(cudaEventSynchronize(morton.stop));
      CUDA_SAFE_CALL(cudaEventSynchronize(sort.stop));
      CUDA_SAFE_CALL(cudaEventSynchronize(build.stop));
      CUDA_SAFE_CALL(cudaEventSynchronize(total.stop));

      timings->scene_bounds_ms = elapsed_ms(scene_bounds);
      timings->morton_ms = elapsed_ms(morton);
      timings->sort_ms = elapsed_ms(sort);
      timings->build_ms = elapsed_ms(build);
      timings->build_pipeline_ms = elapsed_ms(total);
    }
  }  // namespace

  void allocate_workspace(cudaStream_t stream, Workspace& workspace, unsigned int n_faces)
  {
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_scene_bounds, sizeof(AABB), stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_morton_codes, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_morton_codes_sorted, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_cluster_indices, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_cluster_indices_sorted, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_parent_indices, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_cluster_count, sizeof(unsigned int), stream));

    cub::DeviceRadixSort::SortPairs(
        nullptr,
        workspace.sort_temp_storage_bytes,
        workspace.d_morton_codes,
        workspace.d_morton_codes_sorted,
        workspace.d_cluster_indices,
        workspace.d_cluster_indices_sorted,
        n_faces,
        0,
        32,
        stream);
    CUDA_SAFE_CALL(cudaMallocAsync(&workspace.d_sort_temp_storage, workspace.sort_temp_storage_bytes, stream));
  }

  void free_workspace(cudaStream_t stream, Workspace& workspace)
  {
    if (workspace.d_sort_temp_storage != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_sort_temp_storage, stream));
    if (workspace.d_cluster_count != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_cluster_count, stream));
    if (workspace.d_parent_indices != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_parent_indices, stream));
    if (workspace.d_cluster_indices_sorted != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_cluster_indices_sorted, stream));
    if (workspace.d_cluster_indices != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_cluster_indices, stream));
    if (workspace.d_morton_codes_sorted != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_morton_codes_sorted, stream));
    if (workspace.d_morton_codes != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_morton_codes, stream));
    if (workspace.d_scene_bounds != nullptr) CUDA_SAFE_CALL(cudaFreeAsync(workspace.d_scene_bounds, stream));
    workspace = {};
  }

  void build(cudaStream_t stream, unsigned int* d_faces, float* d_vertices, BVHNode* d_nodes, Workspace& workspace, unsigned int n_faces)
  {
    build_impl(stream, d_faces, d_vertices, d_nodes, workspace, n_faces, nullptr);
  }

  void build(
      cudaStream_t stream,
      unsigned int* d_faces,
      float* d_vertices,
      BVHNode* d_nodes,
      Workspace& workspace,
      unsigned int n_faces,
      BuildTimings& timings)
  {
    build_impl(stream, d_faces, d_vertices, d_nodes, workspace, n_faces, &timings);
  }
}  // namespace cuda::nexus_hploc
