#pragma once

#include "aabb.h"
#include "utils/defines.h"

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
struct BVHNode {
  AABB aabb;
  unsigned int left_child_index;
  unsigned int right_child_index;

  __host__ __device__ bool is_leaf() const { return left_child_index == INVALID_INDEX; }
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
static_assert(sizeof(BVHNode) == sizeof(AABB) + 2 * 4, "BVHNode size mismatch");
