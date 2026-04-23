#include <cub/device/device_radix_sort.cuh>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../kernels.h"
#include "../structs/aabb.h"
#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

__device__ __forceinline__ int common_bits_from(MortonCode *morton_codes, int n, int i, int j)
{
  if (i < 0 || j < 0 || i >= n || j >= n) return -1;
  if (morton_codes[i] == morton_codes[j]) return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
  return __clz(morton_codes[i] ^ morton_codes[j]);
}

namespace cuda::lbvh
{
  __global__ void build_bvh_kernel(BVHNode *bvh, MortonCode *morton_codes, unsigned int n_faces)
  {
    int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n_faces - 1) return;

    int diff = common_bits_from(morton_codes, n_faces, index, index + 1) - common_bits_from(morton_codes, n_faces, index, index - 1);

    int direction = (diff > 0) ? 1 : -1;
    int dmin = common_bits_from(morton_codes, n_faces, index, index - direction);

    int r_max = 2;
    while (common_bits_from(morton_codes, n_faces, index, index + r_max * direction) > dmin) r_max *= 2;

    int r = 0;
    for (int t = r_max >> 1; t > 0; t >>= 1) {
      if (common_bits_from(morton_codes, n_faces, index, index + (r + t) * direction) > dmin) r += t;
    }

    r = index + r * direction;
    int dnode = common_bits_from(morton_codes, n_faces, index, r);

    int s = 0;
    for (int t = r_max >> 1; t > 0; t >>= 1) {
      if (common_bits_from(morton_codes, n_faces, index, index + (s + t) * direction) > dnode) s += t;
    }

    int y = index + s * direction + min(direction, 0);

    int min_val = min(index, r);
    int max_val = max(index, r);

    bvh[index].left_child_index = (min_val == y) ? n_faces - 1 + min_val : y;
    bvh[index].right_child_index = (max_val == y + 1) ? n_faces - 1 + max_val : y + 1;
  }

  __global__ void build_aabb_leaves_kernel(BVHNode *bvh, unsigned int *faces, float *vertices, unsigned int *indices, unsigned int n_faces)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;

    unsigned int i = indices[index];

    unsigned int f0 = faces[3 * i + 0];
    unsigned int f1 = faces[3 * i + 1];
    unsigned int f2 = faces[3 * i + 2];

    float3 v0 = {vertices[3 * f0 + 0], vertices[3 * f0 + 1], vertices[3 * f0 + 2]};
    float3 v1 = {vertices[3 * f1 + 0], vertices[3 * f1 + 1], vertices[3 * f1 + 2]};
    float3 v2 = {vertices[3 * f2 + 0], vertices[3 * f2 + 1], vertices[3 * f2 + 2]};

    unsigned int leaf_index = index + n_faces - 1;

    bvh[leaf_index].aabb.min_x = fminf(fminf(v0.x, v1.x), v2.x);
    bvh[leaf_index].aabb.min_y = fminf(fminf(v0.y, v1.y), v2.y);
    bvh[leaf_index].aabb.min_z = fminf(fminf(v0.z, v1.z), v2.z);
    bvh[leaf_index].aabb.max_x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    bvh[leaf_index].aabb.max_y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    bvh[leaf_index].aabb.max_z = fmaxf(fmaxf(v0.z, v1.z), v2.z);
    bvh[leaf_index].left_child_index = NO_NODE_ID;
    bvh[leaf_index].right_child_index = NO_NODE_ID;
  }

  __global__ void build_aabb_kernel(BVHNode *bvh, unsigned int *parents, unsigned int *flags, unsigned int n_faces)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;

    // Начинаем с leaf-узла
    int node = n_faces - 1 + index;

    // Поднимаемся к корню
    while (true) {
      unsigned int p = parents[node];
      if (p == NO_NODE_ID) break;

      // Атомарно: если мы первые - выходим, иначе строим AABB
      unsigned int old = atomicAdd(&flags[p], 1u);
      if (old == 0) break;  // первый поток - ждём второго

      // Второй поток: оба ребёнка готовы
      unsigned int left = bvh[p].left_child_index;
      unsigned int right = bvh[p].right_child_index;

      bvh[p].aabb = AABB::union_of(bvh[left].aabb, bvh[right].aabb);
      node = p;
      __threadfence();
    }
  }

  // Вычисляем parent[] по построенным leftChildIndex/rightChildIndex
  __global__ void compute_parents_kernel(BVHNode *bvh, unsigned int *parents, unsigned int n_faces)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) parents[0] = NO_NODE_ID;
    if (index < n_faces - 1) {
      unsigned int left = bvh[index].left_child_index;
      unsigned int right = bvh[index].right_child_index;
      if (left < 2 * n_faces - 1) parents[left] = index;
      if (right < 2 * n_faces - 1) parents[right] = index;
    }
  }
}  // namespace cuda::lbvh
