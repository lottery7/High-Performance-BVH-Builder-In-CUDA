#pragma once
#include <cstdint>

#define MortonCode uint32_t

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(MortonCode) == 4, "MortonCode must be 32-bit");
