#pragma once

#include "aabb.h"

struct WideBVHNode {
  AABB aabb;
  AABB child_aabbs[8];
  unsigned int child_indices[8];
  unsigned int valid_mask;
  unsigned int internal_mask;
};

static_assert(sizeof(WideBVHNode) == sizeof(AABB) * 9 + sizeof(unsigned int) * 10, "WideBVHNode size mismatch");
static_assert(sizeof(WideBVHNode) == 256, "WideBVHNode must stay tightly packed");
