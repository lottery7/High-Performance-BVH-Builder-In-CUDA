#include <libbase/stats.h>

#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/lbvh/lbvh.cuh"
#include "../kernels/ray_tracing/rt.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "lbvh.h"
#include "utils/device_buffer.h"

#define EXPERIMENT_NAME "LBVH"

using Stage = benchmark::GpuStageProfiler::Stage;

RayTracingResult run_lbvh(cudaStream_t stream, const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene_gpu.n_faces;
  const unsigned int n_nodes = 2 * n_faces - 1;

  DeviceBuffer<BVH2Node> bvh(n_nodes, stream);
  DeviceBuffer<AABB> primitives_aabb(n_faces, stream);
  DeviceBuffer<AABB> scene_aabb(1, stream);
  DeviceBuffer<MortonCode> morton_codes(n_faces, stream);
  DeviceBuffer<MortonCode> morton_codes_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> indices(n_faces, stream);
  DeviceBuffer<unsigned int> indices_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> parents(n_nodes, stream);
  DeviceBuffer<unsigned int> flags(n_faces - 1, stream);

  CUDA_SYNC_STREAM(stream);

  benchmark::GpuStageProfiler prof(stream, benchmark_iters());

  const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
    prof.record_start(Stage::Total);
    prof.record_start(Stage::TotalBuild);

    prof.record_start(Stage::PrimitivesAABB);
    cuda::lbvh::build_primitives_aabb_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene_gpu.d_faces,
        n_faces,
        scene_gpu.d_vertices,
        indices,
        primitives_aabb);
    prof.record_stop(Stage::PrimitivesAABB);

    prof.record_start(Stage::SceneAABB);
    cuda::compute_scene_aabb(stream, primitives_aabb, n_faces, scene_aabb);
    prof.record_stop(Stage::SceneAABB);

    prof.record_start(Stage::MortonCodes);
    cuda::compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene_aabb,
        scene_gpu.d_faces,
        scene_gpu.d_vertices,
        morton_codes,
        n_faces);
    prof.record_stop(Stage::MortonCodes);

    prof.record_start(Stage::Sort);
    cuda::sort_pairs(stream, morton_codes.get(), morton_codes_sorted.get(), indices.get(), indices_sorted.get(), n_faces, 0, 32);
    prof.record_stop(Stage::Sort);

    prof.record_start(Stage::Build);
    cuda::lbvh::build_bvh_kernel<<<compute_grid(2 * n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(
        bvh,
        morton_codes_sorted,
        parents,
        primitives_aabb,
        indices_sorted,
        n_faces);
    prof.record_stop(Stage::Build);

    prof.record_start(Stage::InternalNodesAABB);
    CUDA_SAFE_CALL(cudaMemsetAsync(flags, 0, sizeof(unsigned int) * (n_faces - 1), stream));
    cuda::lbvh::build_internal_nodes_aabb_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(bvh, parents, flags, n_faces);
    prof.record_stop(Stage::InternalNodesAABB);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::lbvh::rt_lbvh_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        bvh,
        indices_sorted,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    prof.record_stop(Stage::RayTracing);

    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect({
          Stage::PrimitivesAABB,
          Stage::SceneAABB,
          Stage::MortonCodes,
          Stage::Sort,
          Stage::Build,
          Stage::InternalNodesAABB,
          Stage::TotalBuild,
          Stage::Total,
          Stage::RayTracing,
      });
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);
  std::cout << EXPERIMENT_NAME " primitives AABB (in ms) - " << prof.median(Stage::PrimitivesAABB) << std::endl;
  std::cout << EXPERIMENT_NAME " scene aabb (in ms) - " << prof.median(Stage::SceneAABB) << std::endl;
  std::cout << EXPERIMENT_NAME " morton codes (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << EXPERIMENT_NAME " sort (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << EXPERIMENT_NAME " build bvh (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << EXPERIMENT_NAME " internal nodes AABB (in ms) - " << prof.median(Stage::InternalNodesAABB) << std::endl;
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

  report_sah(stream, bvh.get(), n_nodes);

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_lbvh", res.face_ids, res.ao);

  CUDA_SYNC_STREAM(stream);

  return res;
}
