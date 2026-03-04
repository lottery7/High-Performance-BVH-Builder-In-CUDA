#pragma once

#include "libimages/images.h"

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};
