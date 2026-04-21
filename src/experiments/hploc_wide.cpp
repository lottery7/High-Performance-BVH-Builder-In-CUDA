#include "hploc_wide.h"

#include <libbase/stats.h>

#include <iostream>
#include <string>
#include <vector>

#include "../kernels/h_ploc/hploc.h"
#include "../kernels/h_ploc/hploc_wide.h"
#include "../kernels/kernels.h"
#include "../kernels/structs/wide_bvh_node.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"

template <unsigned int Arity>
RayTracingResult run_hploc_wide(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  const std::string experiment_name = "H-PLOC BVH" + std::to_string(Arity);

  std::cout << "\n=== Experiment: " << experiment_name << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_binary_nodes = 2 * n_faces - 1;

  BVHNode* d_binary_bvh = nullptr;
  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_cluster_ids = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_n_binary_nodes = nullptr;

  WideBVHNode<Arity>* d_wide_bvh = nullptr;
  unsigned long long* d_tasks = nullptr;
  unsigned int* d_next_task = nullptr;
  unsigned int* d_next_wide_node = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_binary_bvh, sizeof(BVHNode) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids, sizeof(unsigned int) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_binary_nodes, sizeof(unsigned int), stream));

  CUDA_SAFE_CALL(cudaMallocAsync(&d_wide_bvh, sizeof(WideBVHNode<Arity>) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_tasks, sizeof(unsigned long long) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_next_task, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_next_wide_node, sizeof(unsigned int), stream));
  CUDA_SYNC_STREAM(stream);

  std::vector<double> bvh2_build_times;
  std::vector<double> conversion_times;
  std::vector<double> total_build_times;
  CudaEventTimer timer;
  const AdaptiveWarmupResult build_warmup = benchmark::run_adaptive([&](bool collect) {
    const double bvh2_time = timer.measure(stream, [&] {
      cuda::hploc::build(
          stream,
          scene.aabb,
          scene.d_faces,
          scene.d_vertices,
          d_binary_bvh,
          d_parents,
          d_morton_codes,
          d_cluster_ids,
          d_n_binary_nodes,
          n_faces);
    });
    const double conversion_time = timer.measure(stream, [&] {
      cuda::hploc::convert_to_wide<Arity>(stream, d_binary_bvh, d_wide_bvh, d_tasks, d_next_task, d_next_wide_node, n_faces);
    });
    const double total_build_time = bvh2_time + conversion_time;

    if (collect) {
      bvh2_build_times.push_back(bvh2_time);
      conversion_times.push_back(conversion_time);
      total_build_times.push_back(total_build_time);
    }

    return total_build_time;
  });
  print_warmup_report(experiment_name, build_warmup);

  const double build_mtris = n_faces * 1e-6f / stats::median(total_build_times);
  std::cout << experiment_name << " BVH2 build times (in seconds) - " << stats::valuesStatsLine(bvh2_build_times) << std::endl;
  std::cout << experiment_name << " conversion times (in seconds) - " << stats::valuesStatsLine(conversion_times) << std::endl;
  std::cout << experiment_name << " total build times (in seconds) - " << stats::valuesStatsLine(total_build_times) << std::endl;
  std::cout << experiment_name << " total build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, d_next_wide_node, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  std::cout << "Total wide nodes: " << n_wide_nodes << std::endl;
  curassert(0 < n_wide_nodes && n_wide_nodes <= max_binary_nodes, 63123333);

  fb.clear();

  std::vector<double> rt_times;
  const AdaptiveWarmupResult rt_warmup = benchmark::run_adaptive([&](bool collect) {
    const double sample_seconds = timer.measure(stream, [&] {
      cuda::rt_hploc_wide<Arity>(stream, width, height, scene.d_vertices, scene.d_faces, d_wide_bvh, fb.d_face_id, fb.d_ao, scene.d_camera);
    });
    if (collect) rt_times.push_back(sample_seconds);
    return sample_seconds;
  });
  print_warmup_report(experiment_name + " ray tracing", rt_warmup);

  const double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << experiment_name << " ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << experiment_name << " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << experiment_name << " total frame time: " << stats::median(total_build_times) + stats::median(rt_times) << " seconds" << std::endl;

  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" + experiment_name, res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_binary_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_wide_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_tasks, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_next_task, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_next_wide_node, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}

template RayTracingResult run_hploc_wide<4>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
template RayTracingResult run_hploc_wide<8>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
