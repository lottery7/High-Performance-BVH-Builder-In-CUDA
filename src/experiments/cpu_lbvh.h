#pragma once

#include <cuda_runtime_api.h>

#include <filesystem>

#include "../io/scene_reader.h"
#include "../utils/gpu_wrappers.h"
#include "common.h"

RayTracingResult runCPULBVH(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters);
