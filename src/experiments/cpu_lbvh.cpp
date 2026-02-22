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
CPULBVHResult runCPULBVH(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters,
    std::vector<double>& out_perf_mrays)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;

  std::vector<BVHNodeGPU> nodes_cpu;
  std::vector<uint32_t> indices_cpu;

  timer cpu_lbvh_t;
  buildLBVH_CPU(scene.vertices, scene.faces, nodes_cpu, indices_cpu);
  double build_time = cpu_lbvh_t.elapsed();

  std::cout << "CPU build LBVH in " << build_time << " sec" << std::endl;
  std::cout << "CPU LBVH build performance: " << scene_gpu.nfaces * 1e-6f / build_time << " MTris/s" << std::endl;

  LBVHDataGPU lbvh(nodes_cpu, indices_cpu);

  fb.clear(stream);

  std::vector<double> rt_times;
  for (int iter = 0; iter < niters; ++iter) {
    timer t;
    cuda::ray_tracing_render_using_lbvh(
        stream,
        dim3(divCeil(width, 16), divCeil(height, 16)),
        dim3(16, 16),
        scene_gpu.vertices,
        scene_gpu.faces,
        lbvh.nodes,
        lbvh.leaf_face_indices,
        fb.face_id,
        fb.ao,
        scene_gpu.camera,
        scene_gpu.nfaces);
    rt_times.push_back(t.elapsed());
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays << " MRays/s" << std::endl;
  out_perf_mrays.push_back(mrays);

  CPULBVHResult result;
  result.lbvh_build_time = build_time;
  result.rt_time_sum = stats::sum(rt_times);
  fb.readback(result.face_ids, result.ao);
  saveFramebuffers(results_dir, "with_cpu_lbvh", result.face_ids, result.ao);

  // LBVHDataGPU освобождается здесь автоматически (RAII)
  return result;
}
