#pragma once

#include "aabb.h"
#include "utils/defines.h"

template <unsigned int Arity>
struct WideBVHNode {
  AABB aabb;
  unsigned int children[Arity];

  __host__ __device__ bool is_leaf() const { return children[0] == INVALID_INDEX; }
};

using WideBVHNode4 = WideBVHNode<4>;
using WideBVHNode8 = WideBVHNode<8>;

static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
