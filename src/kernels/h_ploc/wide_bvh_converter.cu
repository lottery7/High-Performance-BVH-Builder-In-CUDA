#include "wide_bvh_converter.h"

#include <cfloat>
#include <cuda_runtime.h>

#include "../../utils/defines.h"
#include "../../utils/utils.h"

#define FULL_MASK 0xFFFFFFFFu
#define WARP_SIZE 32
#define INVALID_ASSIGNMENT 0xFu

namespace
{
  __device__ __forceinline__ uint64_t global_load_u64(const uint64_t* addr)
  {
    uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(addr));
    return value;
  }

  __device__ __forceinline__ void global_store_u64(uint64_t* addr, uint64_t value)
  {
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(addr), "l"(value));
  }

  __device__ __forceinline__ unsigned int get_nibble(unsigned int value, unsigned int index) { return (value >> (4 * index)) & 0xFu; }

  __device__ __forceinline__ void set_nibble(unsigned int& value, unsigned int index, unsigned int nibble)
  {
    value &= ~(0xFu << (4 * index));
    value |= nibble << (4 * index);
  }

  __device__ __forceinline__ unsigned int count_bits_below(unsigned int value, unsigned int index)
  {
    const unsigned int mask = (index == 0) ? 0u : ((1u << index) - 1u);
    return __popc(value & mask);
  }

  __device__ __forceinline__ float3 aabb_centroid_sum(const AABB& aabb)
  {
    return make_float3(aabb.min_x + aabb.max_x, aabb.min_y + aabb.max_y, aabb.min_z + aabb.max_z);
  }

  __device__ __forceinline__ bool is_leaf(const BVHNode& node) { return node.left_child_index == INVALID_INDEX; }

  __device__ __forceinline__ float get_slot_cost(unsigned int slot, const float3& offset)
  {
    return ((slot >> 2) & 1u ? -1.0f : 1.0f) * offset.x + ((slot >> 1) & 1u ? -1.0f : 1.0f) * offset.y + (slot & 1u ? -1.0f : 1.0f) * offset.z;
  }

  __host__ __device__ __forceinline__ uint64_t make_work_item(unsigned int bvh2_node_index, unsigned int wide_node_index)
  {
    return (static_cast<uint64_t>(bvh2_node_index) << 32u) | static_cast<uint64_t>(wide_node_index);
  }

  __device__ __forceinline__ void assign_children_to_slots(const BVHNode* bvh2_nodes, const BVHNode& parent, unsigned int child_nodes[8], unsigned int child_count, unsigned int& assignments)
  {
    assignments = 0xFFFFFFFFu;
    const float3 parent_centroid = aabb_centroid_sum(parent.aabb);

    for (unsigned int child = 0; child < child_count; ++child) {
      const float3 child_centroid = aabb_centroid_sum(bvh2_nodes[child_nodes[child]].aabb);
      const float3 offset = make_float3(parent_centroid.x - child_centroid.x, parent_centroid.y - child_centroid.y, parent_centroid.z - child_centroid.z);

      float best_cost = -FLT_MAX;
      unsigned int best_slot = INVALID_ASSIGNMENT;
      for (unsigned int slot = 0; slot < 8; ++slot) {
        if (get_nibble(assignments, slot) != INVALID_ASSIGNMENT) continue;

        const float cost = get_slot_cost(slot, offset);
        if (cost > best_cost) {
          best_cost = cost;
          best_slot = slot;
        }
      }
      set_nibble(assignments, best_slot, child);
    }
  }

  __device__ __forceinline__ WideBVHNode make_empty_wide_node(const AABB& aabb)
  {
    WideBVHNode node{};
    node.aabb = aabb;
    node.valid_mask = 0;
    node.internal_mask = 0;
    for (unsigned int i = 0; i < 8; ++i) {
      node.child_indices[i] = INVALID_INDEX;
    }
    return node;
  }

  __device__ void create_single_leaf_wide_node(unsigned int bvh2_node_index, unsigned int wide_node_index, const BVHNode* bvh2_nodes, WideBVHNode* wide_nodes)
  {
    const BVHNode& leaf = bvh2_nodes[bvh2_node_index];
    WideBVHNode node = make_empty_wide_node(leaf.aabb);
    node.valid_mask = 1u;
    node.child_aabbs[0] = leaf.aabb;
    node.child_indices[0] = leaf.right_child_index;
    wide_nodes[wide_node_index] = node;
  }

  __global__ void convert_to_wide_bvh_kernel(
      const BVHNode* bvh2_nodes,
      WideBVHNode* wide_nodes,
      unsigned int* node_counter,
      uint64_t* work_items,
      unsigned int* work_counter,
      unsigned int* work_alloc_counter,
      unsigned int n_faces)
  {
    const unsigned int lane_id = threadIdx.x & (WARP_SIZE - 1);

    unsigned int work_id = INVALID_INDEX;
    if (lane_id == 0) {
      work_id = atomicAdd(work_counter, WARP_SIZE);
    }
    work_id = __shfl_sync(FULL_MASK, work_id, 0) + lane_id;

    if (n_faces == 1) {
      if (work_id == 0) create_single_leaf_wide_node(0, 0, bvh2_nodes, wide_nodes);
      return;
    }

    bool lane_active = work_id < n_faces;
    while (true) {
      if (__syncthreads_count(lane_active) == 0) break;
      if (!lane_active) continue;

      const uint64_t work_item = global_load_u64(work_items + work_id);
      const unsigned int bvh2_node_index = static_cast<unsigned int>(work_item >> 32u);
      const unsigned int wide_node_index = static_cast<unsigned int>(work_item);

      if (bvh2_node_index == INVALID_INDEX) continue;

      const BVHNode& bvh2_node = bvh2_nodes[bvh2_node_index];
      if (is_leaf(bvh2_node)) {
        if (wide_node_index != INVALID_INDEX) create_single_leaf_wide_node(bvh2_node_index, wide_node_index, bvh2_nodes, wide_nodes);
        lane_active = false;
        continue;
      }

      unsigned int child_nodes[8];
      unsigned int child_count = 0;
      unsigned int internal_mask = 0;
      int split_slot = 0;

      unsigned int left_right_child[2] = {bvh2_node.left_child_index, bvh2_node.right_child_index};
      while (true) {
        const float left_area = bvh2_nodes[left_right_child[0]].aabb.surface_area();
        const float right_area = bvh2_nodes[left_right_child[1]].aabb.surface_area();
        unsigned int first = left_area < right_area ? 0u : 1u;

        for (unsigned int i = 0; i < 2; ++i) {
          const unsigned int target_index = (i == 0) ? static_cast<unsigned int>(split_slot) : child_count;
          const unsigned int child_node_index = left_right_child[first];
          if (!is_leaf(bvh2_nodes[child_node_index])) internal_mask |= 1u << target_index;
          child_nodes[target_index] = child_node_index;
          ++child_count;
          first = 1u - first;
        }

        split_slot = (internal_mask == 0u) ? -1 : 31 - __clz(internal_mask);
        if (split_slot < 0 || child_count == 8) break;

        internal_mask &= ~(1u << split_slot);
        --child_count;

        const unsigned int next_child = child_nodes[split_slot];
        left_right_child[0] = bvh2_nodes[next_child].left_child_index;
        left_right_child[1] = bvh2_nodes[next_child].right_child_index;
      }

      unsigned int assignments = 0xFFFFFFFFu;
      assign_children_to_slots(bvh2_nodes, bvh2_node, child_nodes, child_count, assignments);

      unsigned int reordered_internal_mask = 0;
      unsigned int valid_mask = 0;
      for (unsigned int slot = 0; slot < 8; ++slot) {
        const unsigned int assignment = get_nibble(assignments, slot);
        if (assignment == INVALID_ASSIGNMENT) continue;

        valid_mask |= 1u << slot;
        const bool child_is_internal = (internal_mask >> assignment) & 1u;
        reordered_internal_mask |= static_cast<unsigned int>(child_is_internal) << slot;
      }

      const unsigned int internal_count = __popc(reordered_internal_mask);
      const unsigned int work_base_index = atomicAdd(work_alloc_counter, child_count - 1);
      const unsigned int child_base_index = atomicAdd(node_counter, internal_count);

      WideBVHNode wide_node = make_empty_wide_node(bvh2_node.aabb);
      wide_node.valid_mask = valid_mask;
      wide_node.internal_mask = reordered_internal_mask;

      unsigned int task_ordinal = 0;
      for (unsigned int slot = 0; slot < 8; ++slot) {
        const unsigned int assignment = get_nibble(assignments, slot);
        if (assignment == INVALID_ASSIGNMENT) continue;

        const unsigned int child_node_index = child_nodes[assignment];
        wide_node.child_aabbs[slot] = bvh2_nodes[child_node_index].aabb;

        unsigned int next_wide_node_index = INVALID_INDEX;
        if ((reordered_internal_mask >> slot) & 1u) {
          next_wide_node_index = child_base_index + count_bits_below(reordered_internal_mask, slot);
          wide_node.child_indices[slot] = next_wide_node_index;
        } else {
          wide_node.child_indices[slot] = bvh2_nodes[child_node_index].right_child_index;
        }

        const unsigned int task_index = (task_ordinal == 0) ? work_id : work_base_index + task_ordinal - 1;
        global_store_u64(work_items + task_index, make_work_item(child_node_index, next_wide_node_index));
        ++task_ordinal;
      }

      wide_nodes[wide_node_index] = wide_node;
    }
  }
}  // namespace

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
      unsigned int n_faces)
  {
    const size_t wide_capacity = static_cast<size_t>((4u * n_faces + 5u) / 7u);
    CUDA_SAFE_CALL(cudaMemsetAsync(d_wide_bvh, 0, sizeof(WideBVHNode) * wide_capacity, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_work_items, 0xFF, sizeof(uint64_t) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_work_counter, 0, sizeof(unsigned int), stream));

    const unsigned int one = 1u;
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_node_count, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_alloc_counter, &one, sizeof(one), cudaMemcpyHostToDevice, stream));

    const unsigned int root_index = n_bvh2_nodes - 1;
    const uint64_t root_work_item = make_work_item(root_index, 0u);
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_items, &root_work_item, sizeof(root_work_item), cudaMemcpyHostToDevice, stream));

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>((n_faces + block_size - 1u) / block_size);
    convert_to_wide_bvh_kernel<<<grid_size, block_size, 0, stream>>>(
        d_bvh2,
        d_wide_bvh,
        d_node_count,
        d_work_items,
        d_work_counter,
        d_work_alloc_counter,
        n_faces);
  }
}  // namespace cuda::wide_hploc
