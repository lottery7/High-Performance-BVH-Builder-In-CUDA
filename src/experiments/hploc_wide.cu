#include <libbase/stats.h>

#include <cub/device/device_radix_sort.cuh>
#include <string>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc_bvh2.cuh"
#include "../kernels/hploc/hploc_wide.cuh"
#include "../kernels/ray_tracing/rt_hploc_wide.cuh"
#include "../kernels/structs/wide_bvh_node.h"
#include "../utils/defines.h"
#include "../utils/device_buffer.h"
#include "../utils/utils.h"
#include "../utils/wide_bvh_sah.h"
#include "benchmark.h"
#include "hploc_wide.h"

using Stage = benchmark::GpuStageProfiler::Stage;

namespace
{
  const AABB empty_scene = AABB::neutral();

  template <unsigned int Arity>
  std::string experiment_name()
  {
    static_assert(Arity == 4 || Arity == 8, "Wide H-PLOC experiment supports BVH4/BVH8 only");
    return "H-PLOC WideBVH" + std::to_string(Arity);
  }

  template <unsigned int Arity>
  RayTracingResult run_hploc_wide(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
  {
    const std::string name = experiment_name<Arity>();
    std::cout << "\n=== Experiment: " << name << std::endl;

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

    size_t temp_storage_bytes = 0;
    CUDA_SAFE_CALL(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            morton_codes.get(),
            morton_codes_sorted.get(),
            clusters.get(),
            clusters_sorted.get(),
            n_faces,
            0,
            30,
            stream));
    DeviceBuffer<uint8_t> temp_storage(temp_storage_bytes, stream);

    DeviceBuffer<WideBVHNode<Arity>> wide_nodes(n_nodes_capacity, stream);
    DeviceBuffer<unsigned long long> tasks(n_faces, stream);
    DeviceBuffer<unsigned int> n_tasks(1, stream);
    DeviceBuffer<unsigned int> n_wide_nodes(1, stream);

    CUDA_SYNC_STREAM(stream);

    benchmark::GpuStageProfiler prof(stream, benchmark_iters());

    const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
      const unsigned long long root_task = cuda::hploc::pack_wide_task(n_nodes_capacity - 1, 0);
      constexpr unsigned int one = 1;

      CUDA_SAFE_CALL(cudaMemcpyAsync(n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(scene_aabb, &empty_scene, sizeof(AABB), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemsetAsync(parents, 0xFF, sizeof(unsigned int) * n_nodes_capacity, stream));
      CUDA_SAFE_CALL(cudaMemsetAsync(tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(tasks, &root_task, sizeof(root_task), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(n_tasks, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(n_wide_nodes, &one, sizeof(one), cudaMemcpyHostToDevice, stream));

      prof.record_start(Stage::Total);
      prof.record_start(Stage::TotalBuild);

      prof.record_start(Stage::Leaves);
      cuda::hploc::build_leaves_kernel<<<div_ceil(n_faces, 128 * 3), 128, 0, stream>>>(
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
      CUDA_SAFE_CALL(
          cub::DeviceRadixSort::SortPairs(
              temp_storage.get(),
              temp_storage_bytes,
              morton_codes.get(),
              morton_codes_sorted.get(),
              clusters.get(),
              clusters_sorted.get(),
              n_faces,
              0,
              30,
              stream));
      prof.record_stop(Stage::Sort);

      prof.record_start(Stage::Build);
      cuda::hploc::build_kernel<<<div_ceil(n_faces, 64), 64, 0, stream>>>(
          parents,
          morton_codes_sorted,
          bvh2_nodes,
          clusters_sorted,
          n_clusters,
          n_faces);
      prof.record_stop(Stage::Build);

      prof.record_start(Stage::Conversion);
      cuda::hploc::convert_to_wide_kernel<Arity><<<div_ceil(n_faces, 64), 64, 0, stream>>>(
          bvh2_nodes,
          wide_nodes,
          tasks,
          n_tasks,
          n_wide_nodes,
          n_faces);
      prof.record_stop(Stage::Conversion);

      prof.record_stop(Stage::TotalBuild);

      prof.record_start(Stage::RayTracing);
      cuda::rt_hploc_wide_kernel<Arity><<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
          scene.d_vertices,
          scene.d_faces,
          wide_nodes,
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
    print_warmup_report(name.c_str(), pipeline_warmup);

    const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);

    std::cout << name << " build leaves times (in ms) - " << prof.median(Stage::Leaves) << std::endl;
    std::cout << name << " morton times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
    std::cout << name << " sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
    std::cout << name << " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
    std::cout << name << " conversion times (in ms) - " << prof.median(Stage::Conversion) << std::endl;
    std::cout << name << " build pipeline times (in ms) - " << prof.median(Stage::TotalBuild) << std::endl;
    std::cout << name << " build performance: " << build_mtris << " MTris/s" << std::endl;

    AABB host_aabb;
    CUDA_SAFE_CALL(cudaMemcpyAsync(&host_aabb, scene_aabb.get(), sizeof(AABB), cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);

    std::cout << "Scene AABB: "
              << "Min(" << host_aabb.min_x << ", " << host_aabb.min_y << ", " << host_aabb.min_z << ") "
              << "Max(" << host_aabb.max_x << ", " << host_aabb.max_y << ", " << host_aabb.max_z << ")\n";

    unsigned int n_wide_nodes_host = INVALID_INDEX;
    CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes_host, n_wide_nodes, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);
    std::cout << "Total wide nodes: " << n_wide_nodes_host << std::endl;
    curassert(0 < n_wide_nodes_host && n_wide_nodes_host <= n_nodes_capacity, 63123333);
    wide_bvh_sah::report_hploc_wide_sah<Arity>(stream, wide_nodes, n_wide_nodes_host);

    const double mrays = width * height * AO_SAMPLES * 1e-3 / prof.median(Stage::RayTracing);
    std::cout << name << " ray tracing frame render times (in ms) - " << prof.median(Stage::RayTracing) << std::endl;
    std::cout << name << " ray tracing performance: " << mrays << " MRays/s" << std::endl;
    std::cout << name << " total pipeline times (in ms) - " << prof.median(Stage::Total) << std::endl;

    RayTracingResult res;
    fb.readback(res.face_ids, res.ao);
    save_framebuffers(results_dir, "with_" + name, res.face_ids, res.ao);

    return res;
  }
}  // namespace

RayTracingResult run_hploc_wide4(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  return run_hploc_wide<4>(stream, scene, fb, results_dir);
}

RayTracingResult run_hploc_wide8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  return run_hploc_wide<8>(stream, scene, fb, results_dir);
}
