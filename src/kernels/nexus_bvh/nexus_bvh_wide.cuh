#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "../structs/aabb.h"
#include "../structs/bvh_node.h"

namespace cuda::nexus_bvh_wide
{
  struct AABB {
    float3 bMin;
    float3 bMax;

    __host__ __device__ float Area() const
    {
      const float3 diff = make_float3(bMax.x - bMin.x, bMax.y - bMin.y, bMax.z - bMin.z);
      return diff.x * diff.y + diff.y * diff.z + diff.z * diff.x;
    }
  };

  struct BVH2Node {
    AABB bounds;
    std::uint32_t leftChild;
    std::uint32_t rightChild;
  };

  struct BVH8NodeExplicit {
    float3 p;
    std::uint8_t e[3];
    std::uint8_t imask = 0;
    std::uint32_t childBaseIdx = 0;
    std::uint32_t primBaseIdx = 0;
    std::uint8_t meta[8];
    std::uint8_t qlox[8], qloy[8], qloz[8];
    std::uint8_t qhix[8], qhiy[8], qhiz[8];
  };

  struct BVH8Node {
    float4 p_e_imask;
    float4 childidx_tridx_meta;
    float4 qlox_qloy;
    float4 qloz_qhix;
    float4 qhiy_qhiz;
  };

  struct BVH8BuildState {
    const BVH2Node* bvh2Nodes = nullptr;
    BVH8Node* bvh8Nodes = nullptr;
    std::uint32_t* primIdx = nullptr;
    std::uint32_t primCount = 0;
    std::uint32_t* nodeCounter = nullptr;
    std::uint32_t* leafCounter = nullptr;
    std::uint64_t* indexPairs = nullptr;
    std::uint32_t* workCounter = nullptr;
    std::uint32_t* workAllocCounter = nullptr;
  };

  __global__ void build_bvh8_kernel(BVH8BuildState buildState);
}

static_assert(sizeof(cuda::nexus_bvh_wide::AABB) == sizeof(::AABB), "Nexus wide AABB layout mismatch");
static_assert(sizeof(cuda::nexus_bvh_wide::BVH2Node) == sizeof(BVHNode), "Nexus wide BVH2 node layout mismatch");
static_assert(
    sizeof(cuda::nexus_bvh_wide::BVH8NodeExplicit) == sizeof(cuda::nexus_bvh_wide::BVH8Node),
    "Nexus wide BVH8 node layout mismatch");
