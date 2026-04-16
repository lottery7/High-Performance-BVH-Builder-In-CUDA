#pragma once

#include <driver_types.h>

#include "KittenEngine/includes/modules/Bound.h"
#include "KittenGpuLBVH/lbvh.cuh"
#include "structs/bvh_node.h"
#include "structs/camera.h"
#include "structs/morton_code.h"
#include "structs/wide_bvh_node.h"

namespace cuda
{
  template <typename T>
  void fill(cudaStream_t stream, T* arr, T val, size_t n);

  void rt_lbvh(
      cudaStream_t stream,
      unsigned int width,
      unsigned int height,
      const float* vertices,
      const unsigned int* faces,
      const BVHNode* bvh_nodes,
      const unsigned int* leaf_tri_indices,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera,
      unsigned int n_faces);

  void rt_hploc(
      cudaStream_t stream,
      unsigned int width,
      unsigned int height,
      const float* vertices,
      const unsigned int* faces,
      const BVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera,
      unsigned int n_faces);

  void rt_hploc_wide(
      cudaStream_t stream,
      unsigned int width,
      unsigned int height,
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera);

  void fill_indices(cudaStream_t stream, unsigned int* d_indices, unsigned int n);

  template <typename K, typename V>
  void sort_by_key(cudaStream_t stream, K* d_keys, V* d_vals, size_t n);

  void compute_faces_aabb(
      cudaStream_t stream,
      const float* d_vertices,
      const unsigned int* d_faces,
      Kitten::Bound<3, float>* d_face_bounds,
      unsigned int n_faces);

  void convert_to_bvh_nodes(
      cudaStream_t stream,
      const Kitten::LBVH::node* kitten_internal_nodes,
      const Kitten::LBVH::aabb* kitten_leaf_aabbs,
      const uint32_t* kitten_obj_ids,
      BVHNode* d_bvh_nodes,
      uint32_t* d_leaf_tri_indices,
      unsigned int n_faces);

  void compute_morton_codes(
      cudaStream_t stream,
      AABB scene_aabb,
      unsigned int* d_faces,
      float* d_vertices,
      MortonCode* d_morton_codes,
      unsigned int n_faces);
}  // namespace cuda
