#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>

#include "../experiments/gpu_brute_force.h"
#include "../io/scene_reader.h"
#include "../utils/gpu_wrappers.h"

struct CPULBVHResult {
  image32i face_ids;
  image32f ao;
  double lbvh_build_time = 0.0;
  double rt_time_sum = 0.0;
};

CPULBVHResult runCPULBVH(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters,
    std::vector<double>& out_perf_mrays);
