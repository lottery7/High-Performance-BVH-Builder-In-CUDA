#include <libbase/stats.h>

#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc_bvh2.cuh"
#include "../kernels/ray_tracing/rt.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "hploc_bvh2.h"
#include "utils/device_buffer.h"

#define EXPERIMENT_NAME "H-PLOC BVH2"

using Stage = benchmark::GpuStageProfiler::Stage;
const AABB empty_scene = AABB::neutral();

RayTracingResult run_hploc(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int n_nodes_capacity = 2 * n_faces - 1;

  DeviceBuffer<BVH2Node> nodes(n_nodes_capacity, stream);
  DeviceBuffer<AABB> scene_aabb(1, stream);
  DeviceBuffer<MortonCode> morton_codes(n_faces, stream);
  DeviceBuffer<MortonCode> morton_codes_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> clusters(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> clusters_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> parents(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> n_clusters(1, stream);
  DeviceBuffer<unsigned int> tree_parents(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> rotations_flags(n_nodes_capacity, stream);

  CUDA_SAFE_CALL(cudaFuncSetAttribute(cuda::hploc::build_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(cuda::hploc::build_kernel, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetAttribute(cuda::hploc::build_leaves_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(cuda::hploc::build_leaves_kernel, cudaFuncCachePreferL1));

  CUDA_SYNC_STREAM(stream);

  benchmark::GpuStageProfiler prof(stream, benchmark_iters());

  const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
    prof.record_start(Stage::Total);
    prof.record_start(Stage::TotalBuild);

    prof.record_start(Stage::Leaves);
    CUDA_SAFE_CALL(cudaMemsetAsync(parents, 0xFF, sizeof(unsigned int) * n_nodes_capacity, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(scene_aabb, &empty_scene, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    // TODO Уменьшение сетки в 3 раза + grid stride дает стабильный прирост в 0.06 ms
    cuda::hploc::build_leaves_kernel<<<div_ceil(n_faces, 64) / 3, 64, 0, stream>>>(
        scene.d_faces,
        n_faces,
        scene.d_vertices,
        nodes,
        clusters,
        scene_aabb);
    prof.record_stop(Stage::Leaves);

    prof.record_start(Stage::MortonCodes);
    cuda::compute_morton_codes_kernel<<<div_ceil(n_faces, DEFAULT_GROUP_SIZE), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene_aabb,
        scene.d_faces,
        scene.d_vertices,
        morton_codes,
        n_faces);
    prof.record_stop(Stage::MortonCodes);

    prof.record_start(Stage::Sort);
    cuda::sort_pairs(stream, morton_codes.get(), morton_codes_sorted.get(), clusters.get(), clusters_sorted.get(), n_faces, 0, 30);
    prof.record_stop(Stage::Sort);

    prof.record_start(Stage::Build);
    constexpr size_t build_kernel_block_size = 64;
    cuda::hploc::build_kernel<<<div_ceil(n_faces, build_kernel_block_size), build_kernel_block_size, 0, stream>>>(
        parents,
        morton_codes_sorted,
        nodes,
        clusters_sorted,
        n_clusters,
        n_faces);
    prof.record_stop(Stage::Build);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::hploc::rt_hploc_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        nodes,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera,
        n_faces);
    prof.record_stop(Stage::RayTracing);

    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect({Stage::Leaves, Stage::MortonCodes, Stage::Sort, Stage::Build, Stage::TotalBuild, Stage::RayTracing, Stage::Total});
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);

  std::cout << EXPERIMENT_NAME " build leaves times (in ms) - " << prof.median(Stage::Leaves) << std::endl;
  std::cout << EXPERIMENT_NAME " morton times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << EXPERIMENT_NAME " sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << prof.median(Stage::TotalBuild) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  AABB host_aabb;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&host_aabb, scene_aabb.get(), sizeof(AABB), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);

  std::cout << "Scene AABB: "
            << "Min(" << host_aabb.min_x << ", " << host_aabb.min_y << ", " << host_aabb.min_z << ") "
            << "Max(" << host_aabb.max_x << ", " << host_aabb.max_y << ", " << host_aabb.max_z << ")\n";

  const double mrays = width * height * AO_SAMPLES * 1e-3 / prof.median(Stage::RayTracing);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << prof.median(Stage::RayTracing) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << prof.median(Stage::Total) << std::endl;

  report_sah_hploc(stream, nodes, n_nodes_capacity, n_faces);

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, res.face_ids, res.ao);

  return res;
}
