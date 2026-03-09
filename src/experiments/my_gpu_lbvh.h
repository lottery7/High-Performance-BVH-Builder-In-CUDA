#pragma once

#include <cuda_runtime.h>

#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "common.h"

RayTracingResult run_my_gpu_lbvh(cudaStream_t stream, const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir);