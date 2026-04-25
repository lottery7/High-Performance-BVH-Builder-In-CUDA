#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "../structs/aabb.h"
#include "../structs/bvh_node.h"

namespace cuda::nexus_bvh
{
  struct BuildState {
    AABB* scene_bounds = nullptr;
    BVH2Node* nodes = nullptr;
    unsigned int* cluster_indices = nullptr;
    unsigned int* parent_indices = nullptr;
    unsigned int prim_count = 0;
    unsigned int* cluster_count = nullptr;
  };

  struct BuildTimings {
    double scene_bounds_ms = 0.0;
    double morton_ms = 0.0;
    double sort_ms = 0.0;
    double build_ms = 0.0;
    double build_pipeline_ms = 0.0;
  };

  struct Workspace {
    AABB* d_scene_bounds = nullptr;
    unsigned int* d_morton_codes = nullptr;
    unsigned int* d_morton_codes_sorted = nullptr;
    unsigned int* d_cluster_indices = nullptr;
    unsigned int* d_cluster_indices_sorted = nullptr;
    unsigned int* d_parent_indices = nullptr;
    unsigned int* d_cluster_count = nullptr;
    void* d_sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
  };

  void allocate_workspace(cudaStream_t stream, Workspace& workspace, unsigned int n_faces);

  void free_workspace(cudaStream_t stream, Workspace& workspace);

  __global__ void compute_scene_bounds_kernel(BuildState build_state, const unsigned int* faces, const float* vertices);

  __global__ void compute_morton_codes_kernel(BuildState build_state, unsigned int* morton_codes);

  __global__ void build_bvh2_kernel(BuildState build_state, const unsigned int* morton_codes);

  void build(cudaStream_t stream, unsigned int* d_faces, float* d_vertices, BVH2Node* d_nodes, Workspace& workspace, unsigned int n_faces);

  void build(cudaStream_t stream, unsigned int* d_faces, float* d_vertices, BVH2Node* d_nodes, Workspace& workspace, unsigned int n_faces, BuildTimings& timings);
}  // namespace cuda::nexus_bvh
