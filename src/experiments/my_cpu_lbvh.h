#pragma once

#include <filesystem>

#include "../io/scene_reader.h"
#include "../utils/device_wrappers.h"
#include "common.h"

RayTracingResult run_cpu_lbvh(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneDevice& scene_gpu,
    FramebuffersDevice& fb,
    const std::string& results_dir,
    int n_iters);
