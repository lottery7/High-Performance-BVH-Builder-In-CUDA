#pragma once

#include <cuda_runtime.h>

struct AABB {
  // Minimum corner of the box
  float min_x;
  float min_y;
  float min_z;

  // Maximum corner of the box
  float max_x;
  float max_y;
  float max_z;

  __host__ __device__ float surface_area() const
  {
    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;
    return 2.0 * (dx * dy + dy * dz + dz * dx);
  }

  __host__ __device__ static AABB union_of(const AABB& lhs, const AABB& rhs)
  {
    AABB aabb;
    aabb.min_x = fminf(lhs.min_x, rhs.min_x);
    aabb.min_y = fminf(lhs.min_y, rhs.min_y);
    aabb.min_z = fminf(lhs.min_z, rhs.min_z);
    aabb.max_x = fmaxf(lhs.max_x, rhs.max_x);
    aabb.max_y = fmaxf(lhs.max_y, rhs.max_y);
    aabb.max_z = fmaxf(lhs.max_z, rhs.max_z);
    return aabb;
  }
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(AABB) == 6 * 4, "AABB size mismatch");
