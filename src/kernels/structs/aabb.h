#pragma once

struct AABB {
  // Minimum corner of the box
  float min_x;
  float min_y;
  float min_z;

  // Maximum corner of the box
  float max_x;
  float max_y;
  float max_z;

  float surface_area() const
  {
    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;
    return 2.0 * (dx * dy + dy * dz + dz * dx);
  }
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(AABB) == 6 * 4, "AABB size mismatch");
