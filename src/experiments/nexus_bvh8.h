#pragma once

#include <cuda_runtime.h>

#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "common.h"

RayTracingResult run_nexus_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
