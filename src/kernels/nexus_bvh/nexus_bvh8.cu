#include <device_launch_parameters.h>

#include <cfloat>

#include "nexus_bvh8.cuh"

namespace
{
  constexpr std::uint32_t FULL_MASK = 0xffffffffu;
  constexpr std::uint32_t WARP_SIZE = 32u;
  constexpr std::uint32_t INVALID_ASSIGNMENT = 0xfu;
  constexpr std::uint32_t INVALID_IDX = static_cast<std::uint32_t>(-1);
  constexpr int NQ = 8;
  constexpr float quant_step = 1.0f / static_cast<float>((1 << NQ) - 1);

  __host__ __device__ inline float3 operator+(float3 lhs, float3 rhs)
  {
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
  }

  __host__ __device__ inline float3 operator-(float3 lhs, float3 rhs)
  {
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
  }

  __device__ __forceinline__ float GetCost(std::uint32_t s, float3 offset)
  {
    return ((s >> 2) & 1 ? -1.0f : 1.0f) * offset.x +
           ((s >> 1) & 1 ? -1.0f : 1.0f) * offset.y +
           (s & 1 ? -1.0f : 1.0f) * offset.z;
  }

  __device__ __forceinline__ void GreedyAssignment(
      float3 parentCentroid,
      std::uint32_t childNodes[8],
      std::uint32_t n,
      std::uint32_t& assignments,
      cuda::nexus_bvh_wide::BVH8BuildState buildState)
  {
    assignments = INVALID_IDX;

    for (std::uint32_t c = 0; c < n; c++) {
      float maxCost = -FLT_MAX;
      std::uint32_t bestSlot = INVALID_ASSIGNMENT;

      cuda::nexus_bvh_wide::AABB childBounds = buildState.bvh2Nodes[childNodes[c]].bounds;
      float3 childCentroid = childBounds.bMax + childBounds.bMin;
      float3 offset = parentCentroid - childCentroid;

      for (std::uint32_t s = 0; s < 8; s++) {
        if (((assignments >> (4 * s)) & 0xfu) != INVALID_ASSIGNMENT) continue;

        float cost = GetCost(s, offset);
        if (cost > maxCost) {
          maxCost = cost;
          bestSlot = s;
        }
      }

      assignments &= ~(0xfu << (4 * bestSlot));
      assignments |= c << (4 * bestSlot);
    }
  }

  __device__ __forceinline__ std::uint32_t CeilLog2(float x)
  {
    std::uint32_t ix = __float_as_uint(x);
    std::uint32_t exp = ((ix >> 23) & 0xFF);
    bool isPow2 = (ix & ((1 << 23) - 1)) == 0;
    return exp + !isPow2;
  }

  __device__ __forceinline__ float InvPow2(std::uint8_t eBiased)
  {
    return __uint_as_float(static_cast<std::uint32_t>(254 - eBiased) << 23);
  }

  __device__ __forceinline__ std::uint64_t GlobalLoad(std::uint64_t* addr)
  {
    std::uint64_t value;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(value) : "l"(addr));
    return value;
  }

  __device__ __forceinline__ void GlobalStore(std::uint64_t* addr, std::uint64_t value)
  {
    asm volatile("st.global.cg.u64 [%0], %1;" :: "l"(addr), "l"(value));
  }

  __device__ __forceinline__ std::uint32_t CountBitsBelow(std::uint32_t x, std::uint32_t i)
  {
    std::uint32_t mask = (1u << i) - 1u;
    return __popc(x & mask);
  }

  __device__ __forceinline__ std::uint32_t GetNibble(std::uint32_t x, std::uint32_t i)
  {
    return (x >> (4 * i)) & 0xfu;
  }

  __device__ __forceinline__ cuda::nexus_bvh_wide::BVH8Node CreateBVH8Node(
      cuda::nexus_bvh_wide::AABB bounds,
      std::uint32_t childNodes[8],
      std::uint32_t childBaseIdx,
      std::uint32_t primBaseIdx,
      std::uint32_t assignments,
      std::uint32_t innerMask,
      std::uint32_t leafMask,
      cuda::nexus_bvh_wide::BVH8BuildState buildState)
  {
    cuda::nexus_bvh_wide::BVH8NodeExplicit bvh8Node;

    float3 diagonal = bounds.bMax - bounds.bMin;
    bvh8Node.p = bounds.bMin;
    bvh8Node.e[0] = static_cast<std::uint8_t>(CeilLog2(diagonal.x * quant_step));
    bvh8Node.e[1] = static_cast<std::uint8_t>(CeilLog2(diagonal.y * quant_step));
    bvh8Node.e[2] = static_cast<std::uint8_t>(CeilLog2(diagonal.z * quant_step));
    bvh8Node.imask = 0;
    bvh8Node.childBaseIdx = childBaseIdx;
    bvh8Node.primBaseIdx = primBaseIdx;

    float3 invE = make_float3(InvPow2(bvh8Node.e[0]), InvPow2(bvh8Node.e[1]), InvPow2(bvh8Node.e[2]));

    for (std::uint32_t i = 0; i < 8; i++) {
      bvh8Node.meta[i] = 0;
      std::uint32_t assignment = GetNibble(assignments, i);

      if (innerMask & (1u << i)) {
        bvh8Node.imask |= 1u << i;
        bvh8Node.meta[i] |= 1u << 5;
        bvh8Node.meta[i] |= 24u + i;
      } else if (assignment != INVALID_ASSIGNMENT) {
        bvh8Node.meta[i] |= (1u << 5) | CountBitsBelow(leafMask, i);
      } else {
        continue;
      }

      cuda::nexus_bvh_wide::AABB childBounds = buildState.bvh2Nodes[childNodes[assignment]].bounds;
      bvh8Node.qlox[i] = static_cast<std::uint8_t>(floorf((childBounds.bMin.x - bounds.bMin.x) * invE.x));
      bvh8Node.qloy[i] = static_cast<std::uint8_t>(floorf((childBounds.bMin.y - bounds.bMin.y) * invE.y));
      bvh8Node.qloz[i] = static_cast<std::uint8_t>(floorf((childBounds.bMin.z - bounds.bMin.z) * invE.z));
      bvh8Node.qhix[i] = static_cast<std::uint8_t>(ceilf((childBounds.bMax.x - bounds.bMin.x) * invE.x));
      bvh8Node.qhiy[i] = static_cast<std::uint8_t>(ceilf((childBounds.bMax.y - bounds.bMin.y) * invE.y));
      bvh8Node.qhiz[i] = static_cast<std::uint8_t>(ceilf((childBounds.bMax.z - bounds.bMin.z) * invE.z));
    }

    return *reinterpret_cast<cuda::nexus_bvh_wide::BVH8Node*>(&bvh8Node);
  }

  __device__ inline void CreateBVH8SingleLeaf(std::uint32_t workId, cuda::nexus_bvh_wide::BVH8BuildState buildState)
  {
    if (workId == 0) {
      std::uint32_t childNodes[8];
      childNodes[0] = 0;
      std::uint32_t assignments = 0xfffffff0u;
      cuda::nexus_bvh_wide::BVH2Node bvh2Node = buildState.bvh2Nodes[0];
      atomicAdd(buildState.leafCounter, 1u);
      buildState.primIdx[0] = 0;

      buildState.bvh8Nodes[0] = CreateBVH8Node(bvh2Node.bounds, childNodes, 0, 0, assignments, 0x0, 0x1, buildState);
    }
  }
}  // namespace

namespace cuda::nexus_bvh_wide
{
  __global__ void build_bvh8_kernel(BVH8BuildState buildState)
  {
    std::uint32_t threadWarpId = threadIdx.x & (WARP_SIZE - 1u);
    const BVH2Node* bvh2Nodes = buildState.bvh2Nodes;
    BVH8Node* bvh8Nodes = buildState.bvh8Nodes;

    std::uint32_t workId;
    if (threadWarpId == 0) workId = atomicAdd(buildState.workCounter, WARP_SIZE);

    workId = __shfl_sync(FULL_MASK, workId, 0) + threadWarpId;

    if (buildState.primCount == 1) {
      CreateBVH8SingleLeaf(workId, buildState);
      return;
    }

    bool laneActive = workId < buildState.primCount;

    while (true) {
      if (__syncthreads_count(laneActive) == 0) break;
      if (!laneActive) continue;

      std::uint64_t indexPair = GlobalLoad(&buildState.indexPairs[workId]);
      std::uint32_t bvh2NodeIdx = indexPair >> 32;
      std::uint32_t bvh8NodeIdx = static_cast<std::uint32_t>(indexPair);

      if (bvh2NodeIdx == INVALID_IDX) continue;

      if (bvh2Nodes[bvh2NodeIdx].leftChild == INVALID_IDX) {
        buildState.primIdx[bvh8NodeIdx] = bvh2Nodes[bvh2NodeIdx].rightChild;
        laneActive = false;
        continue;
      }

      std::uint32_t innerMask = 0;
      std::uint32_t childCount = 0;
      std::uint32_t childNodes[8];

      AABB childBounds[2];
      BVH2Node bvh2Node = bvh2Nodes[bvh2NodeIdx];
      std::uint32_t leftRightChild[2] = {bvh2Node.leftChild, bvh2Node.rightChild};
      int msb = 0;

      while (true) {
        childBounds[0] = bvh2Nodes[leftRightChild[0]].bounds;
        childBounds[1] = bvh2Nodes[leftRightChild[1]].bounds;

        bool first = childBounds[0].Area() < childBounds[1].Area() ? 0 : 1;

#pragma unroll
        for (std::uint32_t i = 0; i < 2; i++) {
          std::uint32_t idx = i == 0 ? static_cast<std::uint32_t>(msb) : childCount;

          if (bvh2Nodes[leftRightChild[first]].leftChild != INVALID_IDX) innerMask |= 1u << idx;

          childNodes[idx] = leftRightChild[first];
          childCount++;
          first = !first;
        }

        msb = 31 - __clz(innerMask);

        if (msb < 0 || childCount == 8) break;

        innerMask &= ~(1u << msb);
        childCount--;

        std::uint32_t newIdx = childNodes[msb];
        leftRightChild[0] = bvh2Nodes[newIdx].leftChild;
        leftRightChild[1] = bvh2Nodes[newIdx].rightChild;
      }

      float3 parentCentroid = bvh2Node.bounds.bMin + bvh2Node.bounds.bMax;
      std::uint32_t assignments;
      GreedyAssignment(parentCentroid, childNodes, childCount, assignments, buildState);

      std::uint32_t newInnerMask = 0;
      std::uint32_t leafMask = 0;
      for (std::uint32_t i = 0; i < 8; i++) {
        if (GetNibble(assignments, i) == INVALID_ASSIGNMENT) continue;

        bool bit = (innerMask >> GetNibble(assignments, i)) & 1u;
        newInnerMask |= static_cast<std::uint32_t>(bit) << i;
        leafMask |= static_cast<std::uint32_t>(!bit) << i;
      }
      innerMask = newInnerMask;

      std::uint32_t innerCount = __popc(innerMask);
      std::uint32_t leafCount = childCount - innerCount;

      std::uint32_t childBaseIdx = atomicAdd(buildState.nodeCounter, innerCount);
      std::uint32_t workBaseIdx = atomicAdd(buildState.workAllocCounter, childCount - 1);

      std::uint32_t primBaseIdx = 0;
      if (leafCount > 0) primBaseIdx = atomicAdd(buildState.leafCounter, leafCount);

      for (std::uint32_t i = 0; i < 8; i++) {
        if (GetNibble(assignments, i) == INVALID_ASSIGNMENT) continue;

        std::uint64_t pair = static_cast<std::uint64_t>(childNodes[GetNibble(assignments, i)]) << 32;
        if (innerMask & (1u << i))
          pair |= childBaseIdx + CountBitsBelow(innerMask, i);
        else
          pair |= primBaseIdx + CountBitsBelow(leafMask, i);

        std::uint32_t c = CountBitsBelow(innerMask | leafMask, i);
        std::uint32_t idx = c == 0 ? workId : workBaseIdx + c - 1;
        GlobalStore(&buildState.indexPairs[idx], pair);
      }

      bvh8Nodes[bvh8NodeIdx] = CreateBVH8Node(bvh2Node.bounds, childNodes, childBaseIdx, primBaseIdx, assignments, innerMask, leafMask, buildState);
    }
  }
}  // namespace cuda::nexus_bvh_wide
