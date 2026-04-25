#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include "libimages/images.h"

struct RuntimeConfig {
  int benchmark_iters = 0;
  int warmup_min_seconds = 5;
  int warmup_max_seconds = 60;
  bool disable_warmup = false;
  int cuda_device = 0;
  std::vector<std::string> experiments;
  std::vector<std::string> scenes;
};

inline RuntimeConfig g_runtime_config;

inline RuntimeConfig& runtime_config() { return g_runtime_config; }

inline const RuntimeConfig& runtime_config_const() { return g_runtime_config; }

inline int benchmark_iters() { return runtime_config_const().benchmark_iters; }

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};
