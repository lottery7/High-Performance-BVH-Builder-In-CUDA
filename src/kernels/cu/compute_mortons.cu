#include <cuda_runtime.h>

#include "../../utils/cuda_utils.h"
#include "../../utils/utils.h"
#include "../defines.h"

__device__ static unsigned int expandBits(unsigned int v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ static unsigned int getMortonCode(float x, float y, float z)
{
  unsigned int ix = min(max((int)(x * 1024.0f), 0), 1023);
  unsigned int iy = min(max((int)(y * 1024.0f), 0), 1023);
  unsigned int iz = min(max((int)(z * 1024.0f), 0), 1023);

  unsigned int xx = expandBits(ix);
  unsigned int yy = expandBits(iy);
  unsigned int zz = expandBits(iz);

  return (xx << 2) | (yy << 1) | zz;
}

__global__ void compute_mortons_kernel(
    const unsigned int* faces,
    const float* vertices,
    unsigned int nfaces,
    float cMinX,
    float cMinY,
    float cMinZ,
    float cMaxX,
    float cMaxY,
    float cMaxZ,
    unsigned int* morton_codes)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nfaces) return;

  const float eps = 1e-9f;
  float dx = fmaxf(cMaxX - cMinX, eps);
  float dy = fmaxf(cMaxY - cMinY, eps);
  float dz = fmaxf(cMaxZ - cMinZ, eps);

  unsigned int f0 = faces[3 * index + 0];
  unsigned int f1 = faces[3 * index + 1];
  unsigned int f2 = faces[3 * index + 2];

  float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
  float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
  float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

  float cx = (v0.x + v1.x + v2.x) / 3.0f;
  float cy = (v0.y + v1.y + v2.y) / 3.0f;
  float cz = (v0.z + v1.z + v2.z) / 3.0f;

  float nx = fminf(fmaxf((cx - cMinX) / dx, 0.0f), 1.0f);
  float ny = fminf(fmaxf((cy - cMinY) / dy, 0.0f), 1.0f);
  float nz = fminf(fmaxf((cz - cMinZ) / dz, 0.0f), 1.0f);

  morton_codes[index] = getMortonCode(nx, ny, nz);
}

namespace cuda
{
  void compute_mortons(
      const cudaStream_t& stream,
      const unsigned int* faces,
      const float* vertices,
      unsigned int nfaces,
      float cMinX,
      float cMinY,
      float cMinZ,
      float cMaxX,
      float cMaxY,
      float cMaxZ,
      unsigned int* morton_codes)
  {
    size_t grid = compute_grid(nfaces);
    compute_mortons_kernel<<<grid, DEFAULT_GROUP_SIZE, 0, stream>>>(faces, vertices, nfaces, cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ, morton_codes);
  }
}  // namespace cuda
