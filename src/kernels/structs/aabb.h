#pragma once

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

struct AABB {
  // Minimum corner of the box
  float min_x = 0;
  float min_y = 0;
  float min_z = 0;

  // Maximum corner of the box
  float max_x = 0;
  float max_y = 0;
  float max_z = 0;

  __host__ __device__ __forceinline__ static AABB neutral() { return {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MIN, FLT_MIN, FLT_MIN}; }

  __host__ __device__ __forceinline__ float surface_area() const { return 2.0f * half_area(); }

  __host__ __device__ __forceinline__ float half_area() const
  {
    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;
    return fmaf(dz, dx + dy, dx * dy);
  }

  __host__ __device__ __forceinline__ void union_with(const AABB& rhs)
  {
    min_x = fminf(min_x, rhs.min_x);
    min_y = fminf(min_y, rhs.min_y);
    min_z = fminf(min_z, rhs.min_z);

    max_x = fmaxf(max_x, rhs.max_x);
    max_y = fmaxf(max_y, rhs.max_y);
    max_z = fmaxf(max_z, rhs.max_z);
  }

  __host__ __device__ __forceinline__ static AABB union_of(const AABB& lhs, const AABB& rhs)
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

  __host__ __device__ __forceinline__ static AABB from_triangle(const float3& p0, const float3& p1, const float3& p2)
  {
    AABB aabb;
    aabb.min_x = fminf(p0.x, fminf(p1.x, p2.x));
    aabb.min_y = fminf(p0.y, fminf(p1.y, p2.y));
    aabb.min_z = fminf(p0.z, fminf(p1.z, p2.z));

    aabb.max_x = fmaxf(p0.x, fmaxf(p1.x, p2.x));
    aabb.max_y = fmaxf(p0.y, fmaxf(p1.y, p2.y));
    aabb.max_z = fmaxf(p0.z, fmaxf(p1.z, p2.z));
    return aabb;
  }
};

__host__ __device__ __forceinline__ float half_surface_area_of_union(const AABB& lhs, const AABB& rhs)
{
  const float min_x = fminf(lhs.min_x, rhs.min_x);
  const float min_y = fminf(lhs.min_y, rhs.min_y);
  const float min_z = fminf(lhs.min_z, rhs.min_z);

  const float max_x = fmaxf(lhs.max_x, rhs.max_x);
  const float max_y = fmaxf(lhs.max_y, rhs.max_y);
  const float max_z = fmaxf(lhs.max_z, rhs.max_z);

  const float dx = max_x - min_x;
  const float dy = max_y - min_y;
  const float dz = max_z - min_z;

  return fmaf(dz, dx + dy, dx * dy);
}

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(AABB) == 6 * 4, "AABB size mismatch");
