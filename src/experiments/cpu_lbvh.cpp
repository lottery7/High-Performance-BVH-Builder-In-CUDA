#pragma once

#include "cpu_lbvh.h"

#include <cuda_runtime_api.h>
#include <libbase/stats.h>
#include <libbase/timer.h>

#include <filesystem>

#include "../cpu_helpers/build_bvh_cpu.h"
#include "../io/scene_reader.h"
#include "../kernels/defines.h"
#include "../kernels/kernels.h"
#include "../utils/cuda_utils.h"
#include "../utils/gpu_wrappers.h"
#include "../utils/utils.h"

// ---------------------------------------------
// Experiment: GPU Ray Tracing with CPU LBVH
// ---------------------------------------------
RayTracingResult runCPULBVH(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const int nfaces = scene.faces.size();

  std::vector<BVHNodeGPU> h_nodes;
  std::vector<uint32_t> h_indices;

  timer cpu_lbvh_t;
  buildLBVH_CPU(scene.vertices, scene.faces, h_nodes, h_indices);
  double build_time = cpu_lbvh_t.elapsed();

  std::cout << "CPU LBVH build took " << build_time << " seconds" << std::endl;
  std::cout << "CPU LBVH build performance: " << nfaces * 1e-6f / build_time << " MTris/s" << std::endl;
  report_sah_h(h_nodes);

  BVHNodeGPU* d_lbvh_nodes = nullptr;
  unsigned int* d_sorted_indices = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_lbvh_nodes, sizeof(BVHNodeGPU) * (2 * nfaces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sorted_indices, sizeof(unsigned int) * nfaces, stream));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_lbvh_nodes, h_nodes.data(), h_nodes.size() * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_sorted_indices, h_indices.data(), h_indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK_STREAM(stream);

  fb.clear();

  std::vector<double> rt_times;
  for (int iter = 0; iter < niters + WARMUP_ITERS; ++iter) {
    cuda::CudaTimer cuda_timer(stream);

    cuda::ray_tracing_render_using_bvh(
        stream,
        width,
        height,
        scene_gpu.vertices,
        scene_gpu.faces,
        d_lbvh_nodes,
        d_sorted_indices,
        fb.face_id,
        fb.ao,
        scene_gpu.camera,
        scene_gpu.nfaces);

    if (iter < WARMUP_ITERS) {
      CUDA_CHECK_STREAM(stream);
      continue;
    }

    rt_times.push_back(cuda_timer.elapsed());
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << "GPU with CPU LBVH total frame time: " << build_time + stats::median(rt_times) << " seconds" << std::endl;

  RayTracingResult result;
  fb.readback(result.face_ids, result.ao);
  saveFramebuffers(results_dir, "with_cpu_lbvh", result.face_ids, result.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_sorted_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_lbvh_nodes, stream));
  CUDA_CHECK_STREAM(stream);

  return result;
}
