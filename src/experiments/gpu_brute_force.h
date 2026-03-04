#pragma once

#include "../utils/gpu_wrappers.h"
#include "common.h"
#include "libimages/images.h"

RayTracingResult runBruteForce(cudaStream_t stream, const SceneGPU& scene_gpu, const FramebuffersGPU& fb, const std::string& results_dir, int niters);
