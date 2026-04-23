#pragma once

#include <cuda_runtime.h>

#include "../structs/bvh_node.h"
#include "../structs/camera.h"

__global__ void rt_hploc_kernel(
    const float* vertices,
    const unsigned int* faces,
    const BVHNode* bvh_nodes,
    int* face_id,
    float* ambient_occlusion,
    const CameraView* camera,
    unsigned int n_faces);
