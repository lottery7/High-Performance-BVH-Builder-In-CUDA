#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "KittenEngine/includes/modules/Bound.h"
#include "KittenGpuLBVH/lbvh.cuh"

// AABB из двух дочерних bounds (объединение)
__device__ __forceinline__ AABB merge_kitten_bounds(const Kitten::Bound<3, float>& b0, const Kitten::Bound<3, float>& b1)
{
  AABB aabb;
  aabb.min_x = fminf(b0.min.x, b1.min.x);
  aabb.min_y = fminf(b0.min.y, b1.min.y);
  aabb.min_z = fminf(b0.min.z, b1.min.z);
  aabb.max_x = fmaxf(b0.max.x, b1.max.x);
  aabb.max_y = fmaxf(b0.max.y, b1.max.y);
  aabb.max_z = fmaxf(b0.max.z, b1.max.z);
  return aabb;
}

__device__ __forceinline__ AABB convert_bound(const Kitten::Bound<3, float>& b)
{
  AABB aabb;
  aabb.min_x = b.min.x;
  aabb.min_y = b.min.y;
  aabb.min_z = b.min.z;
  aabb.max_x = b.max.x;
  aabb.max_y = b.max.y;
  aabb.max_z = b.max.z;
  return aabb;
}

// Kitten: n_faces - 1 внутренних узлов, n_faces листьев
// Наш BVH: [0 .. n_faces-2] - внутренние, [n_faces-1 .. 2*n_faces-2] - листья
// leafStart = n_faces - 1
//
// Kitten leftIdx/rightIdx:
//   MSB=1 -> лист,  индекс в objIDs = value & 0x7FFFFFFF
//   MSB=0 -> внутренний узел, индекс в d_nodes = value
//
// decode: внутренний -> остаётся как есть (индекс в нашем массиве совпадает)
//         лист       -> leafStart + (value & 0x7FFFFFFF)

__device__ __forceinline__ int decode_child(uint32_t child, int leafStart)
{
  if (child & 0x80000000u) {
    // лист: индекс в массиве листьев
    int leafIdx = static_cast<int>(child & 0x7fffffffu);
    return leafStart + leafIdx;
  }
  // внутренний узел
  return static_cast<int>(child);
}

__global__ void convert_to_bvh_nodes_kernel(
    const Kitten::LBVH::node* kitten_internal_nodes,  // размер: n_faces - 1
    const Kitten::LBVH::aabb* kitten_leaf_aabbs,      // размер: n_faces (d_objs)
    const uint32_t* kitten_obj_ids,                   // размер: n_faces (d_objIDs, отсортированный)
    BVHNode* bvh_nodes,                               // размер: 2*n_faces - 1
    uint32_t* leaf_tri_indices,                       // размер: n_faces
    unsigned int n_faces)
{
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = 2u * n_faces - 1u;
  if (idx >= total) return;

  const int leafStart = static_cast<int>(n_faces) - 1;

  if (idx < n_faces - 1u) {
    // Внутренний узел
    const Kitten::LBVH::node& k = kitten_internal_nodes[idx];

    // AABB узла = объединение AABB левого и правого поддерева
    bvh_nodes[idx].aabb = merge_kitten_bounds(k.bounds[0], k.bounds[1]);

    bvh_nodes[idx].left_child_index = decode_child(k.leftIdx, leafStart);
    bvh_nodes[idx].right_child_index = decode_child(k.rightIdx, leafStart);
  } else {
    // Лист
    // leafIdx — позиция в отсортированном массиве листьев
    const int leafIdx = static_cast<int>(idx) - leafStart;

    // Оригинальный ID треугольника (до сортировки по Morton)
    const uint32_t triIdx = kitten_obj_ids[leafIdx];

    // AABB листа берём из d_objs по оригинальному ID
    bvh_nodes[idx].aabb = convert_bound(kitten_leaf_aabbs[triIdx]);

    // Дочерних нет — можно оставить как -1 (или не трогать, рендерер не читает)
    bvh_nodes[idx].left_child_index = -1;
    bvh_nodes[idx].right_child_index = -1;

    // Записываем соответствие: лист leafIdx → треугольник triIdx
    leaf_tri_indices[leafIdx] = triIdx;
  }
}

namespace cuda
{
  void convert_to_bvh_nodes(
      cudaStream_t stream,
      const Kitten::LBVH::node* kitten_internal_nodes,
      const Kitten::LBVH::aabb* kitten_leaf_aabbs,
      const uint32_t* kitten_obj_ids,
      BVHNode* d_bvh_nodes,
      uint32_t* d_leaf_tri_indices,
      unsigned int n_faces)
  {
    convert_to_bvh_nodes_kernel<<<compute_grid(2 * n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(
        kitten_internal_nodes,
        kitten_leaf_aabbs,
        kitten_obj_ids,
        d_bvh_nodes,
        d_leaf_tri_indices,
        n_faces);
  }
}  // namespace cuda
