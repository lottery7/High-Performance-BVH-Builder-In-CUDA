#pragma once

#include "libimages/images.h"

#define WARMUP_ITERS 2
#define BENCHMARK_ITERS 2

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};
