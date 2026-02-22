#pragma once

typedef struct AABBGPU {
  // Minimum corner of the box
  float min_x;
  float min_y;
  float min_z;

  // Maximum corner of the box
  float max_x;
  float max_y;
  float max_z;
} AABBGPU;

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(AABBGPU) == 6 * 4, "AABBGPU size mismatch");
