#pragma once

#include "../utils/device_wrappers.h"
#include "common.h"

RayTracingResult run_kitten_lbvh(const SceneDevice& scene_gpu, FramebuffersDevice& fb, const std::string& results_dir, int n_iters);