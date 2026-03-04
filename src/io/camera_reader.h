#pragma once

#include <string>

#include "../kernels/structs/camera_gpu.h"

// Parse file containing <VCGCamera .../> and <ViewSettings .../>
CameraViewGPU loadViewState(const std::string& path);

// Parse from in-memory XML string (useful for tests)
CameraViewGPU parseViewStateFromString(const std::string& xml);

std::string dumpViewStateToString(const CameraViewGPU& camera);
