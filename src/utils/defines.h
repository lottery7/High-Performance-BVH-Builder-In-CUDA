#pragma once

#define BVH_STACK_SIZE 64

#define DEFAULT_GROUP_SIZE 256
#define DEFAULT_GROUP_SIZE_X 16
#define DEFAULT_GROUP_SIZE_Y 16
#define DEFAULT_GROUP_SIZE_2D dim3(DEFAULT_GROUP_SIZE_X, DEFAULT_GROUP_SIZE_Y)

#define MAX_GRID_SIZE 65535
#define MAX_GRID_SIZE_X 256
#define MAX_GRID_SIZE_Y 256

#define AO_SAMPLES 8
#define NO_FACE_ID (-1)
#define NO_AMBIENT_OCCLUSION (-1.0f)
#define INVALID_INDEX 0xFFFFFFFFu

#define RASSERT_ENABLED 0  // enable for debug, disable before performance evaluation/profiling/commiting
