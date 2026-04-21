#include "my_cpu_lbvh.h"

#include <cuda_runtime_api.h>
#include <libbase/stats.h>
#include <libbase/timer.h>

#include <filesystem>
#include <iostream>

#include "../io/scene_reader.h"
#include "../kernels/kernels.h"
#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "lbvh_cpu.h"

#define EXPERIMENT_NAME "My CPU LBVH"

RayTracingResult run_cpu_lbvh(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const cuda::Scene& scene_gpu,
    const cuda::Framebuffers& fb,
    const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const int n_faces = scene.faces.size();

  std::vector<BVHNode> h_nodes;
  std::vector<uint32_t> h_indices;

  timer cpu_lbvh_t;
  buildLBVH_CPU(scene.vertices, scene.faces, h_nodes, h_indices);
  double build_time = cpu_lbvh_t.elapsed();

  std::cout << EXPERIMENT_NAME " build took " << build_time << " seconds" << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << n_faces * 1e-6f / build_time << " MTris/s" << std::endl;
  report_sah(h_nodes);

  BVHNode* d_lbvh_nodes = nullptr;
  unsigned int* d_sorted_indices = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_lbvh_nodes, sizeof(BVHNode) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sorted_indices, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_lbvh_nodes, h_nodes.data(), h_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_sorted_indices, h_indices.data(), h_indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
  CUDA_SYNC_STREAM(stream);

  fb.clear();
  const int warmup = warmup_iters();
  const int benchmark = benchmark_iters();

  std::vector<double> rt_times;
  for (int iter = 0; iter < benchmark + warmup; ++iter) {
    timer ray_tracing_t;

    cuda::rt_lbvh(
        stream,
        width,
        height,
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        d_lbvh_nodes,
        d_sorted_indices,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    CUDA_SYNC_STREAM(stream);

    if (iter >= warmup) {
      rt_times.push_back(ray_tracing_t.elapsed());
    }
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total frame time: " << build_time + stats::median(rt_times) << " seconds" << std::endl;

  RayTracingResult result;
  fb.readback(result.face_ids, result.ao);
  save_framebuffers(results_dir, "with_cpu_lbvh", result.face_ids, result.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_sorted_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_lbvh_nodes, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
