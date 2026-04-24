#include <cuda_runtime.h>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../structs/aabb.h"
#include "../structs/morton_code.h"
#include "helpers.cuh"

namespace cuda
{

  __global__ void compute_mortons_kernel(AABB scene_aabb, unsigned int *faces, float *vertices, MortonCode *morton_codes, unsigned int n_faces)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;

    const float eps = 1e-9f;
    float dx = fmaxf(scene_aabb.max_x - scene_aabb.min_x, eps);
    float dy = fmaxf(scene_aabb.max_y - scene_aabb.min_y, eps);
    float dz = fmaxf(scene_aabb.max_z - scene_aabb.min_z, eps);

    unsigned int f0 = faces[3 * index + 0];
    unsigned int f1 = faces[3 * index + 1];
    unsigned int f2 = faces[3 * index + 2];

    float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
    float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
    float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

    const float min_x = fminf(fminf(v0.x, v1.x), v2.x);
    const float min_y = fminf(fminf(v0.y, v1.y), v2.y);
    const float min_z = fminf(fminf(v0.z, v1.z), v2.z);
    const float max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    const float max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    const float max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);

    const float cx = 0.5f * (min_x + max_x);
    const float cy = 0.5f * (min_y + max_y);
    const float cz = 0.5f * (min_z + max_z);

    float nx = fminf(fmaxf((cx - scene_aabb.min_x) / dx, 0.0f), 1.0f);
    float ny = fminf(fmaxf((cy - scene_aabb.min_y) / dy, 0.0f), 1.0f);
    float nz = fminf(fmaxf((cz - scene_aabb.min_z) / dz, 0.0f), 1.0f);

    morton_codes[index] = get_morton_code(nx, ny, nz);
  }

}  // namespace cuda
