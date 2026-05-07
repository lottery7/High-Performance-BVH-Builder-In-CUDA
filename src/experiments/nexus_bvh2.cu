#include <cfloat>
#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/nexus_bvh/nexus_bvh2.cuh"
#include "../utils/defines.h"
#include "../utils/device_buffer.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "kernels/hploc/hploc_bvh2.cuh"
#include "kernels/ray_tracing/rt_bvh2.cuh"
#include "nexus_bvh2.h"

#define EXPERIMENT_NAME "NexusBVH BVH2"

using Stage = benchmark::GpuStageProfiler::Stage;

RayTracingResult run_nexus_bvh2(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_nodes = 2 * n_faces - 1;

  BVH2Node* d_bvh = nullptr;
  cuda::nexus_bvh::Workspace workspace;
  DeviceBuffer<float> ao_radius(1, stream);

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVH2Node) * max_nodes, stream));
  cuda::nexus_bvh::allocate_workspace(stream, workspace, n_faces);
  CUDA_SYNC_STREAM(stream);

  cuda::nexus_bvh::BuildState build_state{};
  build_state.scene_bounds = workspace.d_scene_bounds;
  build_state.nodes = d_bvh;
  build_state.cluster_indices = workspace.d_cluster_indices;
  build_state.parent_indices = workspace.d_parent_indices;
  build_state.prim_count = n_faces;
  build_state.cluster_count = workspace.d_cluster_count;

  const AABB empty_scene_bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

  benchmark::GpuStageProfiler prof(stream, benchmark_iters());

  const AdaptiveWarmupResult build_warmup = benchmark::run_adaptive([&](bool collect) {
    build_state.cluster_indices = workspace.d_cluster_indices;
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.scene_bounds, &empty_scene_bounds, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(build_state.parent_indices, 0xff, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.cluster_count, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
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
    constexpr unsigned int block_size = 64;
    prof.record_start(Stage::Build);
    cuda::nexus_bvh::build_bvh2_kernel<<<div_ceil(static_cast<int>(n_faces), static_cast<int>(block_size)), block_size, 0, stream>>>(
        build_state,
        workspace.d_morton_codes_sorted);
    prof.record_stop(Stage::Build);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::compute_ao_radius(stream, build_state.scene_bounds, ao_radius);
    cuda::rt_bvh2_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        d_bvh,
        max_nodes - 1,
        fb.d_ao,
        ao_radius,
        scene.d_camera);
    prof.record_stop(Stage::RayTracing);
    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect({Stage::SceneAABB, Stage::MortonCodes, Stage::Sort, Stage::Build, Stage::TotalBuild, Stage::RayTracing, Stage::Total});
    }
    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, build_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);
  std::cout << EXPERIMENT_NAME " compute scene bounds times (in ms) - " << prof.median(Stage::SceneAABB) << std::endl;
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << prof.median(Stage::TotalBuild) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_nodes, workspace.d_cluster_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(n_nodes == max_nodes, 226786314);
  report_sah_hploc(stream, d_bvh, max_nodes, n_faces);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / prof.median(Stage::RayTracing);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << prof.median(Stage::RayTracing) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << prof.median(Stage::Total) << std::endl;

  RayTracingResult result;
  fb.readback(result.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, result.ao);

  cuda::nexus_bvh::free_workspace(stream, workspace);
  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
