#pragma once

#include <cuda_runtime.h>

#include <string>

#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "common.h"

RayTracingResult run_hploc_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
