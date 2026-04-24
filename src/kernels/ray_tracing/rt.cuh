#pragma once

#include <cuda_runtime.h>

#include "../nexus_bvh/nexus_bvh_wide.cuh"
#include "../structs/bvh_node.h"
#include "../structs/camera.h"

namespace cuda
{
  namespace hploc
  {
    __global__ void rt_hploc_kernel(
        const float* vertices,
        const unsigned int* faces,
        const BVHNode* bvh_nodes,
        int* face_id,
        float* ambient_occlusion,
        const CameraView* camera,
        unsigned int n_faces);

    template <unsigned int Arity>
    __global__ void rt_hploc_wide_kernel(
        const float* vertices,
        const unsigned int* faces,
        const WideBVHNode<Arity>* bvh_nodes,
        int* face_id,
        float* ambient_occlusion,
        const CameraView* camera);

    __global__ void rt_nexus_bvh_wide_kernel(
        const float* vertices,
        const unsigned int* faces,
        const cuda::nexus_bvh_wide::BVH8Node* bvh_nodes,
        const unsigned int* prim_idx,
        int* face_id,
        float* ambient_occlusion,
        const CameraView* camera);
  }  // namespace hploc

  namespace lbvh
  {
    __global__ void rt_lbvh_kernel(
        const float* vertices,
        const unsigned int* faces,
        const BVHNode* bvh_nodes,
        const unsigned int* leaf_tri_indices,
        int* face_id,
        float* ambient_occlusion,
        const CameraView* camera,
        unsigned int n_faces);
  }  // namespace lbvh

}  // namespace cuda
