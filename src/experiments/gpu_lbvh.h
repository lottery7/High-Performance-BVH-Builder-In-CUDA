#pragma once

#include "../utils/gpu_wrappers.h"
#include "common.h"

RayTracingResult runGPULBVH(cudaStream_t stream, const SceneGPU& scene_gpu, FramebuffersGPU& fb, const std::string& results_dir, int niters);