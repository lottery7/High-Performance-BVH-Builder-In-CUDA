#pragma once

#include "../utils/gpu_wrappers.h"
#include "libimages/images.h"

struct BruteForceResult {
  image32i face_ids;
  image32f ao;
  double pcie_read_time = 0.0;
  double total_time = 0.0;
};

BruteForceResult runBruteForce(cudaStream_t stream, const SceneGPU& scene_gpu, const FramebuffersGPU& fb, const std::string& results_dir, int niters);
