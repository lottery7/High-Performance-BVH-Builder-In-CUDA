#pragma once

#include "aabb.h"

template <unsigned int Arity>
struct WideBVHNode {
  AABB aabb;
  AABB child_aabbs[Arity];
  unsigned int child_indices[Arity];
  unsigned int valid_mask;
  unsigned int primitive_mask;
};

using WideBVHNode4 = WideBVHNode<4>;
using WideBVHNode8 = WideBVHNode<8>;

static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
