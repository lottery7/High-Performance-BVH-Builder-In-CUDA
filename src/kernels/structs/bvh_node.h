#pragma once

#include "aabb.h"
#include "utils/defines.h"

struct BVH2Node {
  AABB aabb;
  unsigned int left_child_index;
  unsigned int right_child_index;

  __host__ __device__ bool is_leaf() const { return left_child_index == INVALID_INDEX; }
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
static_assert(sizeof(BVH2Node) == sizeof(AABB) + 2 * 4, "BVH2Node size mismatch");
