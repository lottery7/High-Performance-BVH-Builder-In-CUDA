#include <libbase/stats.h>

#include <cub/device/device_radix_sort.cuh>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc_bvh2.cuh"
#include "../kernels/hploc/hploc_wide.cuh"
#include "../kernels/ray_tracing/rt.cuh"
#include "../kernels/structs/wide_bvh_node.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "../utils/wide_bvh_sah.h"
#include "benchmark.h"
#include "hploc_wide.h"
#include "utils/device_buffer.h"

using Stage = benchmark::GpuStageProfiler::Stage;

template <unsigned int Arity>
RayTracingResult run_hploc_wide(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  const std::string experiment_name = "H-PLOC BVH" + std::to_string(Arity);

  std::cout << "\n=== Experiment: " << experiment_name << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int n_nodes_capacity = 2 * n_faces - 1;
  const unsigned int binary_bvh_root_index = 2 * n_faces - 2;
  const unsigned long long root_task = cuda::hploc::pack_task(binary_bvh_root_index, 0);
  constexpr unsigned int one = 1;

  DeviceBuffer<BVH2Node> nodes(n_nodes_capacity, stream);
  DeviceBuffer<AABB> primitives_aabb(n_faces, stream);
  DeviceBuffer<AABB> scene_aabb(1, stream);
  DeviceBuffer<MortonCode> morton_codes(n_faces, stream);
  DeviceBuffer<MortonCode> morton_codes_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> clusters(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> clusters_sorted(n_faces, stream);
  DeviceBuffer<unsigned int> parents(n_nodes_capacity, stream);
  DeviceBuffer<unsigned int> n_clusters(1, stream);

  DeviceBuffer<WideBVHNode<Arity>> wide_bvh(n_nodes_capacity, stream);
  DeviceBuffer<unsigned long long> tasks(n_faces, stream);
  DeviceBuffer<unsigned int> next_task(1, stream);
  DeviceBuffer<unsigned int> next_wide_node(1, stream);

  CUDA_SYNC_STREAM(stream);

  benchmark::GpuStageProfiler prof(stream, benchmark_iters());

  const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
    prof.record_start(Stage::Total);
    prof.record_start(Stage::TotalBuild);

    prof.record_start(Stage::Leaves);
    CUDA_SAFE_CALL(cudaMemsetAsync(parents, INVALID_INDEX, sizeof(unsigned int) * n_nodes_capacity, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));
    cuda::hploc::build_leaves_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene.d_faces,
        n_faces,
        scene.d_vertices,
        nodes,
        primitives_aabb,
        clusters);
    cuda::fill_indices(stream, clusters, n_faces);
    prof.record_stop(Stage::Leaves);

    prof.record_start(Stage::SceneAABB);
    cuda::compute_scene_aabb(stream, primitives_aabb, n_faces, scene_aabb);
    prof.record_stop(Stage::SceneAABB);

    prof.record_start(Stage::MortonCodes);
    cuda::compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene_aabb,
        scene.d_faces,
        scene.d_vertices,
        morton_codes,
        n_faces);
    prof.record_stop(Stage::MortonCodes);

    prof.record_start(Stage::Sort);
    cuda::sort_pairs(stream, morton_codes.get(), morton_codes_sorted.get(), clusters.get(), clusters_sorted.get(), n_faces, 2, 32);
    prof.record_stop(Stage::Sort);

    prof.record_start(Stage::Build);
    constexpr size_t block_size = 128;
    cuda::hploc::build_kernel<<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(
        parents,
        morton_codes_sorted,
        nodes,
        clusters_sorted,
        n_clusters,
        n_faces);
    prof.record_stop(Stage::Build);

    prof.record_start(Stage::Conversion);
    CUDA_SAFE_CALL(cudaMemsetAsync(tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(tasks, &root_task, sizeof(root_task), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(next_task, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(next_wide_node, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    cuda::hploc::convert_to_wide_kernel<Arity>
        <<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(nodes, wide_bvh, tasks, next_task, next_wide_node, n_faces);
    prof.record_stop(Stage::Conversion);

    prof.record_stop(Stage::TotalBuild);

    prof.record_start(Stage::RayTracing);
    cuda::hploc::rt_hploc_wide_kernel<Arity><<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        wide_bvh,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera);
    prof.record_stop(Stage::RayTracing);

    prof.record_stop(Stage::Total);
    prof.cuda_sync_event(Stage::Total);

    const double total_ms = prof.elapsed_ms(Stage::Total);

    if (collect) {
      prof.collect(
          {Stage::Leaves,
           Stage::SceneAABB,
           Stage::MortonCodes,
           Stage::Sort,
           Stage::Build,
           Stage::Conversion,
           Stage::TotalBuild,
           Stage::RayTracing,
           Stage::Total});
    }

    return total_ms;
  });
  print_warmup_report(experiment_name, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / prof.median(Stage::TotalBuild);

  std::cout << experiment_name << " leaves times (in ms) - " << prof.median(Stage::Leaves) << std::endl;
  std::cout << experiment_name << " scene aabb times (in ms) - " << prof.median(Stage::SceneAABB) << std::endl;
  std::cout << experiment_name << " morton times (in ms) - " << prof.median(Stage::MortonCodes) << std::endl;
  std::cout << experiment_name << " sort times (in ms) - " << prof.median(Stage::Sort) << std::endl;
  std::cout << experiment_name << " build kernel times (in ms) - " << prof.median(Stage::Build) << std::endl;
  std::cout << experiment_name << " conversion times (in ms) - " << prof.median(Stage::Conversion) << std::endl;
  std::cout << experiment_name << " build pipeline times (in ms) - " << prof.median(Stage::TotalBuild) << std::endl;
  std::cout << experiment_name << " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, next_wide_node, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_wide_nodes && n_wide_nodes <= n_nodes_capacity, 63123333);
  wide_bvh_sah::report_hploc_wide_sah(stream, wide_bvh.get(), n_wide_nodes);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / prof.median(Stage::RayTracing);
  std::cout << experiment_name << " ray tracing frame render times (in ms) - " << prof.median(Stage::RayTracing) << std::endl;
  std::cout << experiment_name << " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << experiment_name << " total pipeline times (in ms) - " << prof.median(Stage::Total) << std::endl;

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" + experiment_name, res.face_ids, res.ao);

  return res;
}

template RayTracingResult run_hploc_wide<4>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
template RayTracingResult run_hploc_wide<8>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
