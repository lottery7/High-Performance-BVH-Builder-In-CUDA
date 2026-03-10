#pragma once

#include "libimages/images.h"

#define WARMUP_ITERS 0
#define BENCHMARK_ITERS 1

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};
