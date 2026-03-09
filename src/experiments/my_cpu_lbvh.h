#pragma once

#include <filesystem>

#include "../io/scene_reader.h"
#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "common.h"

RayTracingResult run_cpu_lbvh(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const cuda::Scene& scene_gpu,
    const cuda::Framebuffers& fb,
    const std::string& results_dir);
