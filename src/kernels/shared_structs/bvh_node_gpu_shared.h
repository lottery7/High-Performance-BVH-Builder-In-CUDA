#pragma once

#include <cstdint>

#include "aabb_gpu_shared.h"

// POD AABB struct with identical layout in C++ / CUDA / OpenCL / Vulkan C-like code.
// Uses only scalar floats to avoid float3/vector alignment differences.
typedef struct BVHNodeGPU {
  AABBGPU aabb;
  uint32_t leftChildIndex;
  uint32_t rightChildIndex;
} BVHNodeGPU;

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(uint32_t) == 4, "GPUC_UINT must be 32-bit");
static_assert(sizeof(BVHNodeGPU) == sizeof(AABBGPU) + 2 * 4, "BVHNodeGPU size mismatch");
