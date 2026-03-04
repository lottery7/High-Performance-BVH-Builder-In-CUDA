#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>

#include "../experiments/gpu_brute_force.h"
#include "../utils/gpu_wrappers.h"

RayTracingResult runGPULBVH(cudaStream_t stream, const SceneGPU& scene_gpu, FramebuffersGPU& fb, const std::string& results_dir, int niters);