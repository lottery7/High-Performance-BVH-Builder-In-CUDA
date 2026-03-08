#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "../defines.h"
#include "../structs/bvh_node.h"

// Каждый leaf-поток "поднимается" вверх по дереву.
// Первый из двух детей, достигший узла, выходит (atomic CAS возвращает 0).
// Второй - строит AABB и продолжает вверх.
__global__ void build_aabb_kernel(
    unsigned int nfaces,
    BVHNode* lbvh,
    int* parent,          // parent[i] = родитель узла i (-1 для корня)
    unsigned int* flags)  // флаги посещаемости, изначально 0
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nfaces) return;

  // Начинаем с leaf-узла
  int node = nfaces - 1 + index;

  // Поднимаемся к корню
  while (true) {
    int p = parent[node];
    if (p < 0) break;  // достигли корня

    // Атомарно: если мы первые - выходим, иначе строим AABB
    unsigned int old = atomicAdd(&flags[p], 1u);
    if (old == 0u) break;  // первый поток - ждём второго

    // Второй поток: оба ребёнка готовы
    int left = lbvh[p].left_child_index;
    int right = lbvh[p].right_child_index;

    AABB& la = lbvh[left].aabb;
    AABB& ra = lbvh[right].aabb;

    lbvh[p].aabb.min_x = fminf(la.min_x, ra.min_x);
    lbvh[p].aabb.min_y = fminf(la.min_y, ra.min_y);
    lbvh[p].aabb.min_z = fminf(la.min_z, ra.min_z);
    lbvh[p].aabb.max_x = fmaxf(la.max_x, ra.max_x);
    lbvh[p].aabb.max_y = fmaxf(la.max_y, ra.max_y);
    lbvh[p].aabb.max_z = fmaxf(la.max_z, ra.max_z);

    node = p;
    __threadfence();  // видимость записей для других потоков
  }
}

// Вычисляем parent[] по построенным leftChildIndex/rightChildIndex
__global__ void compute_parents_kernel(unsigned int n_faces, const BVHNode* lbvh, int* parent)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Root
  if (index == 0) parent[0] = -1;

  // Internal nodes: 0 .. nfaces-2
  if (index < n_faces - 1) {
    unsigned int left = lbvh[index].left_child_index;
    unsigned int right = lbvh[index].right_child_index;
    if (left < 2 * n_faces - 1) parent[left] = index;
    if (right < 2 * n_faces - 1) parent[right] = index;
  }
}

namespace cuda
{
  void build_aabb(cudaStream_t stream, unsigned int n_faces, BVHNode* d_lbvh, int* d_parent, unsigned int* d_flags)
  {
    compute_parents_kernel<<<compute_grid(n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(n_faces, d_lbvh, d_parent);
    cudaMemsetAsync(d_flags, 0, sizeof(unsigned int) * (n_faces - 1), stream);
    build_aabb_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(n_faces, d_lbvh, d_parent, d_flags);
  }
}  // namespace cuda
