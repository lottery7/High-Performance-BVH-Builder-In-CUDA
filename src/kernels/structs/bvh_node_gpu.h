#pragma once

#include "aabb_gpu.h"

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
typedef struct BVHNodeGPU {
  AABBGPU aabb;
  unsigned int leftChildIndex;
  unsigned int rightChildIndex;
} BVHNodeGPU;

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
static_assert(sizeof(BVHNodeGPU) == sizeof(AABBGPU) + 2 * 4, "BVHNodeGPU size mismatch");
