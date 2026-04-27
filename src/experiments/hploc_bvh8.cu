#include <libbase/stats.h>

#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc_bvh2.cuh"
#include "../kernels/ray_tracing/rt_bvh2.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "hploc_bvh8.h"
#include "kernels/hploc/hploc_bvh8.cuh"
#include "kernels/ray_tracing/rt_bvh8.cuh"
#include "kernels/ray_tracing/rt_compressed_bvh8.cuh"
#include "utils/device_buffer.h"
#include "utils/wide_bvh_sah.h"

#define EXPERIMENT_NAME "H-PLOC BVH8"

using Stage = benchmark::GpuStageProfiler::Stage;
const AABB empty_scene = AABB::neutral();

RayTracingResult run_hploc_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int n_nodes_capacity = 2 * n_faces - 1;

  DeviceBuffer<BVH2Node> bvh2_nodes(n_nodes_capacity, stream);
  DeviceBuffer<AABB> scene_aabb(1, stream);
  DeviceBuffer<MortonCode> morton_codes(n_faces, stream);
  DeviceBuffer<MortonCode> morton_codes_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> clusters(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> clusters_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> parents(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> n_clusters(1, stream);

  DeviceBuffer<BVH8Node> bvh8_nodes(n_nodes_capacity, stream);
  DeviceBuffer<unsigned long long> tasks(n_faces, stream);
  DeviceBuffer<unsigned int> n_tasks(1, stream);
  DeviceBuffer<unsigned int> n_bvh8_nodes(1, stream);

  DeviceBuffer<unsigned int> bvh8_prim_indices(n_faces, stream);
  DeviceBuffer<unsigned int> n_bvh8_leaves(1, stream);

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
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(scene_aabb, &empty_scene, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    cuda::hploc::build_leaves_kernel<<<div_ceil(n_faces, 128) / 3, 128, 0, stream>>>(
        scene.d_faces,
        n_faces,
        scene.d_vertices,
        bvh2_nodes,
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
    CUDA_SAFE_CALL(cudaMemsetAsync(parents, 0xFF, sizeof(unsigned int) * n_nodes_capacity, stream));
    cuda::hploc::build_kernel<<<div_ceil(n_faces, 64), 64, 0, stream>>>(
        parents,
        morton_codes_sorted,
        bvh2_nodes,
        clusters_sorted,
        n_clusters,
        n_faces);
    prof.record_stop(Stage::Build);

    prof.record_start(Stage::Conversion);
    const unsigned long long root_task = cuda::hploc::pack_task(n_nodes_capacity - 1, 0);
    constexpr unsigned int one = 1;
    constexpr unsigned int zero = 0;
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_bvh8_leaves, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(tasks, &root_task, sizeof(unsigned long long), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_tasks, &one, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_bvh8_nodes, &one, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    cuda::hploc::build_bvh8_kernel<<<div_ceil(n_faces, 64), 64, 0, stream>>>(
        bvh2_nodes,
        bvh8_nodes,
        bvh8_prim_indices,
        tasks,
        n_tasks,
        n_bvh8_nodes,
        n_bvh8_leaves,
        n_faces);
    prof.record_stop(Stage::Conversion);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::rt_bvh8_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        bvh8_nodes,
        bvh8_prim_indices,
        0,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera);
    prof.record_stop(Stage::RayTracing);

    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect(
          {Stage::Leaves, Stage::MortonCodes, Stage::Sort, Stage::Build, Stage::Conversion, Stage::TotalBuild, Stage::RayTracing, Stage::Total});
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);

  std::cout << EXPERIMENT_NAME " build leaves times (in ms) - " << prof.median(Stage::Leaves) << std::endl;
  std::cout << EXPERIMENT_NAME " morton times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << EXPERIMENT_NAME " sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << EXPERIMENT_NAME " conversion times (in ms) - " << prof.median(Stage::Conversion) << std::endl;
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

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, n_bvh8_nodes, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_wide_nodes && n_wide_nodes <= n_nodes_capacity, 226786315);
  wide_bvh_sah::report_nexus_bvh8_sah(stream, reinterpret_cast<cuda::nexus_bvh_wide::BVH8Node*>(bvh8_nodes.get()), n_wide_nodes);

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, res.face_ids, res.ao);

  return res;
}
