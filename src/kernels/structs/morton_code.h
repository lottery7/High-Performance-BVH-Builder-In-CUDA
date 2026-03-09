#pragma once

#define MortonCode unsigned int

__host__ __device__ __forceinline__ unsigned int clamp_0_1023(int v) { return (unsigned int)(v < 0 ? 0 : (v > 1023 ? 1023 : v)); }

__host__ __device__ __forceinline__ unsigned int expand_bits(unsigned int v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__host__ __device__ __forceinline__ MortonCode get_morton_code(float x, float y, float z)
{
  unsigned int ix = clamp_0_1023((int)(x * 1024.0f));
  unsigned int iy = clamp_0_1023((int)(y * 1024.0f));
  unsigned int iz = clamp_0_1023((int)(z * 1024.0f));

  unsigned int xx = expand_bits(ix);
  unsigned int yy = expand_bits(iy);
  unsigned int zz = expand_bits(iz);

  return (xx << 2) | (yy << 1) | zz;
}

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(MortonCode) == 4, "MortonCode must be 32-bit");
