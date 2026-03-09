#pragma once

#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "common.h"

RayTracingResult run_kitten_lbvh(const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir);