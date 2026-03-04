#pragma once

#include "libimages/images.h"

#define WARMUP_ITERS 10

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};
