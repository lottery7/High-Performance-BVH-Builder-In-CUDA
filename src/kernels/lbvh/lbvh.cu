#include <cub/device/device_radix_sort.cuh>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/helpers.cuh"
#include "../structs/aabb.h"
#include "../structs/bvh_node.h"
#include "../structs/morton_code.h"

__device__ __forceinline__ int common_bits_from(MortonCode* morton_codes, int n, int i, int j)
{
  if (i < 0 || j < 0 || i >= n || j >= n) return -1;
  if (morton_codes[i] == morton_codes[j]) return 32 + __clz(static_cast<unsigned int>(i) ^ static_cast<unsigned int>(j));
  return __clz(morton_codes[i] ^ morton_codes[j]);
}

namespace cuda::lbvh
{
  __global__ void build_bvh_kernel(
      BVH2Node* __restrict__ nodes,
      MortonCode* __restrict__ morton_codes,
      unsigned int* __restrict__ parents,
      const AABB* __restrict__ primitives_aabb,
      const unsigned int* __restrict__ primitives,
      unsigned int n_faces)
  {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) parents[0] = INVALID_INDEX;
    if (index >= 2 * n_faces - 1) return;
    if (index >= n_faces - 1) {
      const unsigned int leaf_index = index - (n_faces - 1);
      unsigned int primitive_index = primitives[leaf_index];
      BVH2Node node = {primitives_aabb[primitive_index], INVALID_INDEX, primitive_index};
      nodes[index] = node;
      return;
    }

    const int left_common = common_bits_from(morton_codes, n_faces, index, index - 1);
    const int right_common = common_bits_from(morton_codes, n_faces, index, index + 1);

    const int direction = (right_common > left_common) ? 1 : -1;
    const int dmin = (right_common > left_common) ? left_common : right_common;

    int r_max = 2;
    while (common_bits_from(morton_codes, n_faces, index, index + r_max * direction) > dmin) r_max *= 2;

    int r = 0;
    for (int t = r_max >> 1; t > 0; t >>= 1) {
      if (common_bits_from(morton_codes, n_faces, index, index + (r + t) * direction) > dmin) r += t;
    }

    r = index + r * direction;
    const int dnode = common_bits_from(morton_codes, n_faces, index, r);

    int s = 0;
    for (int t = r_max >> 1; t > 0; t >>= 1) {
      if (common_bits_from(morton_codes, n_faces, index, index + (s + t) * direction) > dnode) s += t;
    }

    const int y = index + s * direction + min(direction, 0);
    const int min_val = min(index, r);
    const int max_val = max(index, r);

    BVH2Node node{};
    node.left_child_index = (min_val == y) ? n_faces - 1 + min_val : y;
    node.right_child_index = (max_val == y + 1) ? n_faces - 1 + max_val : y + 1;
    nodes[index] = node;
    parents[node.left_child_index] = index;
    parents[node.right_child_index] = index;
  }

  __global__ void build_primitives_aabb_kernel(
      const unsigned int* __restrict__ faces,
      unsigned int n_faces,
      const float* __restrict__ vertices,
      unsigned int* __restrict__ primitive_indices,
      AABB* __restrict__ primitives_aabb)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;
    uint3 face = load_face(faces, index);
    float3 v0 = load_vertex(vertices, face.x);
    float3 v1 = load_vertex(vertices, face.y);
    float3 v2 = load_vertex(vertices, face.z);
    primitives_aabb[index] = AABB::from_triangle(v0, v1, v2);
    primitive_indices[index] = index;
  }

  __global__ void build_internal_nodes_aabb_kernel(
      BVH2Node* __restrict__ nodes,
      unsigned int* __restrict__ parents,
      unsigned int* __restrict__ flags,
      unsigned int n_faces)
  {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_faces) return;

    // Начинаем с leaf-узла
    int node = n_faces - 1 + index;

    // Поднимаемся к корню
    while (true) {
      unsigned int p = parents[node];
      if (p == INVALID_INDEX) break;

      // Атомарно: если мы первые - выходим, иначе строим AABB
      unsigned int old = atomicAdd(&flags[p], 1u);
      if (old == 0) break;  // первый поток - ждём второго

      // Второй поток: оба ребёнка готовы
      unsigned int left = nodes[p].left_child_index;
      unsigned int right = nodes[p].right_child_index;

      nodes[p].aabb = AABB::union_of(nodes[left].aabb, nodes[right].aabb);
      node = p;
      __threadfence();
    }
  }
}  // namespace cuda::lbvh
