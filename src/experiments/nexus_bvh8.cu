#include <cfloat>
#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/nexus_bvh/nexus_bvh2.cuh"
#include "../kernels/nexus_bvh/nexus_bvh8.cuh"
#include "../utils/defines.h"
#include "../utils/device_buffer.h"
#include "../utils/utils.h"
#include "../utils/wide_bvh_sah.h"
#include "benchmark.h"
#include "kernels/ray_tracing/rt_bvh8.cuh"
#include "nexus_bvh8.h"

#define EXPERIMENT_NAME "NexusBVH BVH8"

using Stage = benchmark::GpuStageProfiler::Stage;

RayTracingResult run_nexus_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_binary_nodes = 2 * n_faces - 1;
  const unsigned int max_wide_nodes = static_cast<unsigned int>(div_ceil(static_cast<int>(4u * n_faces - 1u), 7));

  BVH2Node* d_binary_bvh = nullptr;
  cuda::nexus_bvh::Workspace workspace;
  cuda::nexus_bvh_wide::BVH8Node* d_wide_bvh = nullptr;
  unsigned int* d_prim_idx = nullptr;
  unsigned int* d_node_counter = nullptr;
  unsigned int* d_leaf_counter = nullptr;
  unsigned int* d_work_counter = nullptr;
  unsigned int* d_work_alloc_counter = nullptr;
  std::uint64_t* d_index_pairs = nullptr;
  DeviceBuffer<float> ao_radius(1, stream);

  CUDA_SAFE_CALL(cudaMallocAsync(&d_binary_bvh, sizeof(BVH2Node) * max_binary_nodes, stream));
  cuda::nexus_bvh::allocate_workspace(stream, workspace, n_faces);
  CUDA_SAFE_CALL(cudaMallocAsync(&d_wide_bvh, sizeof(cuda::nexus_bvh_wide::BVH8Node) * max_wide_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_prim_idx, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_node_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_leaf_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_alloc_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_index_pairs, sizeof(std::uint64_t) * n_faces, stream));
  CUDA_SYNC_STREAM(stream);

  cuda::nexus_bvh::BuildState build_state{};
  build_state.scene_bounds = workspace.d_scene_bounds;
  build_state.nodes = d_binary_bvh;
  build_state.cluster_indices = workspace.d_cluster_indices;
  build_state.parent_indices = workspace.d_parent_indices;
  build_state.prim_count = n_faces;
  build_state.cluster_count = workspace.d_cluster_count;

  cuda::nexus_bvh_wide::BVH8BuildState wide_build_state{};
  wide_build_state.bvh2Nodes = reinterpret_cast<const cuda::nexus_bvh_wide::BVH2Node*>(d_binary_bvh);
  wide_build_state.bvh8Nodes = d_wide_bvh;
  wide_build_state.primIdx = d_prim_idx;
  wide_build_state.primCount = n_faces;
  wide_build_state.nodeCounter = d_node_counter;
  wide_build_state.leafCounter = d_leaf_counter;
  wide_build_state.indexPairs = d_index_pairs;
  wide_build_state.workCounter = d_work_counter;
  wide_build_state.workAllocCounter = d_work_alloc_counter;

  const AABB empty_scene_bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  const std::uint64_t first_pair = static_cast<std::uint64_t>(max_binary_nodes - 1) << 32u;
  const unsigned int zero = 0;
  const unsigned int one = 1;

  benchmark::GpuStageProfiler prof(stream, benchmark_iters());

  const AdaptiveWarmupResult build_warmup = benchmark::run_adaptive([&](bool collect) {
    build_state.cluster_indices = workspace.d_cluster_indices;
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.scene_bounds, &empty_scene_bounds, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(build_state.parent_indices, 0xff, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.cluster_count, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_index_pairs, 0xFF, sizeof(std::uint64_t) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_index_pairs, &first_pair, sizeof(first_pair), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_node_counter, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_alloc_counter, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_counter, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_leaf_counter, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));
    fb.clear();

    prof.record_start(Stage::Total);
    prof.record_start(Stage::TotalBuild);

    prof.record_start(Stage::SceneAABB);
    cuda::nexus_bvh::compute_scene_bounds_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        build_state,
        scene.d_faces,
        scene.d_vertices);
    prof.record_stop(Stage::SceneAABB);

    prof.record_start(Stage::MortonCodes);
    cuda::nexus_bvh::compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, workspace.d_morton_codes);
    prof.record_stop(Stage::MortonCodes);

    prof.record_start(Stage::Sort);
    cub::DeviceRadixSort::SortPairs(
        workspace.d_sort_temp_storage,
        workspace.sort_temp_storage_bytes,
        workspace.d_morton_codes,
        workspace.d_morton_codes_sorted,
        workspace.d_cluster_indices,
        workspace.d_cluster_indices_sorted,
        n_faces,
        0,
        32,
        stream);
    prof.record_stop(Stage::Sort);

    build_state.cluster_indices = workspace.d_cluster_indices_sorted;
    constexpr int binary_block_size = 64;
    prof.record_start(Stage::Build);
    cuda::nexus_bvh::build_bvh2_kernel<<<div_ceil(n_faces, binary_block_size), binary_block_size, 0, stream>>>(
        build_state,
        workspace.d_morton_codes_sorted);
    prof.record_stop(Stage::Build);

    prof.record_start(Stage::Conversion);
    constexpr int wide_block_size = 256;
    cuda::nexus_bvh_wide::build_bvh8_kernel<<<div_ceil(n_faces, wide_block_size), wide_block_size, 0, stream>>>(wide_build_state);
    prof.record_stop(Stage::Conversion);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::compute_ao_radius(stream, build_state.scene_bounds, ao_radius);
    cuda::rt_bvh8_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        reinterpret_cast<BVH8Node*>(d_wide_bvh),
        d_prim_idx,
        0,
        fb.d_ao,
        ao_radius,
        scene.d_camera);
    prof.record_stop(Stage::RayTracing);

    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect(
          {Stage::SceneAABB, Stage::MortonCodes, Stage::Sort, Stage::Build, Stage::Conversion, Stage::TotalBuild, Stage::RayTracing, Stage::Total});
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, build_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);
  std::cout << EXPERIMENT_NAME " compute scene bounds times (in ms) - " << prof.median(Stage::SceneAABB) << std::endl;
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << EXPERIMENT_NAME " conversion times (in ms) - " << prof.median(Stage::Conversion) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << prof.median(Stage::TotalBuild) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, d_node_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_wide_nodes && n_wide_nodes <= max_wide_nodes, 226786315);
  wide_bvh_sah::report_nexus_bvh8_sah(stream, d_wide_bvh, n_wide_nodes);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / prof.median(Stage::RayTracing);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << prof.median(Stage::RayTracing) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << prof.median(Stage::Total) << std::endl;

  RayTracingResult result;
  fb.readback(result.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, result.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_index_pairs, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_alloc_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_leaf_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_node_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_prim_idx, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_wide_bvh, stream));
  cuda::nexus_bvh::free_workspace(stream, workspace);
  CUDA_SAFE_CALL(cudaFreeAsync(d_binary_bvh, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
