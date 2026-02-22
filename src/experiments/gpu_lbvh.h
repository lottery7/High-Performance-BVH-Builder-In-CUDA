#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>

#include "../experiments/gpu_brute_force.h"
#include "../utils/gpu_wrappers.h"

void runGPULBVH(
    cudaStream_t stream,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters,
    std::vector<double>& out_perf_mrays,
    std::vector<double>& out_build_mtris,
    double& out_build_time_sum,
    double& out_rt_time_sum,
    const image32f& bf_ao,
    const image32i& bf_face_ids,
    bool has_brute_force);