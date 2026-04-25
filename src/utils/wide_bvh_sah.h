#pragma once

#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "../kernels/nexus_bvh/nexus_bvh8.cuh"
#include "../kernels/structs/aabb.h"
#include "../kernels/structs/wide_bvh_node.h"
#include "utils.h"

namespace wide_bvh_sah
{
  struct Costs {
    float traversal = 2.0f;
    float intersection = 3.0f;
  };

  inline AABB empty_aabb()
  {
    const float max_value = std::numeric_limits<float>::max();
    return {max_value, max_value, max_value, -max_value, -max_value, -max_value};
  }

  inline bool is_empty(const AABB& aabb)
  {
    return aabb.min_x > aabb.max_x || aabb.min_y > aabb.max_y || aabb.min_z > aabb.max_z;
  }

  inline unsigned int count_bits_below(unsigned int bits, unsigned int slot)
  {
    const unsigned int mask = slot == 0 ? 0u : ((1u << slot) - 1u);
    return static_cast<unsigned int>(__builtin_popcount(bits & mask));
  }

  inline float uint_bits_to_float(unsigned int bits)
  {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  inline AABB decode_child_aabb(const cuda::nexus_bvh_wide::BVH8NodeExplicit& node, unsigned int slot)
  {
    const float ex = uint_bits_to_float(static_cast<unsigned int>(node.e[0]) << 23);
    const float ey = uint_bits_to_float(static_cast<unsigned int>(node.e[1]) << 23);
    const float ez = uint_bits_to_float(static_cast<unsigned int>(node.e[2]) << 23);

    return {
        node.p.x + ex * node.qlox[slot],
        node.p.y + ey * node.qloy[slot],
        node.p.z + ez * node.qloz[slot],
        node.p.x + ex * node.qhix[slot],
        node.p.y + ey * node.qhiy[slot],
        node.p.z + ez * node.qhiz[slot],
    };
  }

  template <unsigned int Arity>
  float compute_hploc_wide_sah(const std::vector<WideBVHNode<Arity>>& nodes, unsigned int node_count, Costs costs = {})
  {
    rassert(node_count > 0 && node_count <= nodes.size(), 872345101, node_count, nodes.size());

    float sah = 0.0f;
    std::vector<unsigned int> stack{0};

    while (!stack.empty()) {
      const unsigned int node_index = stack.back();
      stack.pop_back();

      rassert(node_index < node_count, 872345102, node_index, node_count);
      const WideBVHNode<Arity>& node = nodes[node_index];
      sah += costs.traversal * node.aabb.surface_area();

      for (unsigned int slot = 0; slot < Arity; ++slot) {
        if ((node.valid_mask & (1u << slot)) == 0u) continue;

        const AABB& child_aabb = node.child_aabbs[slot];
        if ((node.primitive_mask & (1u << slot)) != 0u) {
          sah += costs.intersection * child_aabb.surface_area();
        } else {
          const unsigned int child_index = node.child_indices[slot];
          rassert(child_index < node_count, 872345103, child_index, node_count);
          stack.push_back(child_index);
        }
      }
    }

    const float root_area = nodes[0].aabb.surface_area();
    rassert(root_area > 0.0f, 872345104, root_area);
    return sah / root_area;
  }

  template <unsigned int Arity>
  void report_hploc_wide_sah(const std::vector<WideBVHNode<Arity>>& nodes, unsigned int node_count, Costs costs = {})
  {
    const float sah = compute_hploc_wide_sah(nodes, node_count, costs);
    std::cout << "Wide SAH = " << sah << " (C_trav=" << costs.traversal << ", C_isect=" << costs.intersection << ")" << std::endl;
  }

  template <unsigned int Arity>
  void report_hploc_wide_sah(cudaStream_t stream, const WideBVHNode<Arity>* d_nodes, unsigned int node_count, Costs costs = {})
  {
    std::vector<WideBVHNode<Arity>> nodes(node_count);
    CUDA_SAFE_CALL(cudaMemcpyAsync(nodes.data(), d_nodes, sizeof(WideBVHNode<Arity>) * node_count, cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);
    report_hploc_wide_sah(nodes, node_count, costs);
  }

  inline float compute_nexus_bvh8_sah(
      const std::vector<cuda::nexus_bvh_wide::BVH8Node>& nodes,
      unsigned int node_count,
      Costs costs = {})
  {
    rassert(node_count > 0 && node_count <= nodes.size(), 872345105, node_count, nodes.size());

    float sah = 0.0f;
    AABB root_bounds = empty_aabb();
    std::vector<unsigned int> stack{0};

    while (!stack.empty()) {
      const unsigned int node_index = stack.back();
      stack.pop_back();

      rassert(node_index < node_count, 872345106, node_index, node_count);
      const auto& node = reinterpret_cast<const cuda::nexus_bvh_wide::BVH8NodeExplicit&>(nodes[node_index]);

      AABB node_bounds = empty_aabb();
      for (unsigned int slot = 0; slot < 8; ++slot) {
        if (node.meta[slot] == 0) continue;

        const AABB child_aabb = decode_child_aabb(node, slot);
        node_bounds = is_empty(node_bounds) ? child_aabb : AABB::union_of(node_bounds, child_aabb);

        if ((node.imask & (1u << slot)) != 0u) {
          const unsigned int child_index = node.childBaseIdx + count_bits_below(node.imask, slot);
          rassert(child_index < node_count, 872345107, child_index, node_count);
          stack.push_back(child_index);
        } else {
          sah += costs.intersection * child_aabb.surface_area();
        }
      }

      rassert(!is_empty(node_bounds), 872345108, node_index);
      if (node_index == 0) root_bounds = node_bounds;
      sah += costs.traversal * node_bounds.surface_area();
    }

    const float root_area = root_bounds.surface_area();
    rassert(root_area > 0.0f, 872345109, root_area);
    return sah / root_area;
  }

  inline void report_nexus_bvh8_sah(
      const std::vector<cuda::nexus_bvh_wide::BVH8Node>& nodes,
      unsigned int node_count,
      Costs costs = {})
  {
    const float sah = compute_nexus_bvh8_sah(nodes, node_count, costs);
    std::cout << "Wide SAH = " << sah << " (C_trav=" << costs.traversal << ", C_isect=" << costs.intersection << ")" << std::endl;
  }

  inline void report_nexus_bvh8_sah(
      cudaStream_t stream,
      const cuda::nexus_bvh_wide::BVH8Node* d_nodes,
      unsigned int node_count,
      Costs costs = {})
  {
    std::vector<cuda::nexus_bvh_wide::BVH8Node> nodes(node_count);
    CUDA_SAFE_CALL(cudaMemcpyAsync(nodes.data(), d_nodes, sizeof(cuda::nexus_bvh_wide::BVH8Node) * node_count, cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);
    report_nexus_bvh8_sah(nodes, node_count, costs);
  }
}  // namespace wide_bvh_sah
