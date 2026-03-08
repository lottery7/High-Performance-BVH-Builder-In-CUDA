#pragma once

#include <string>

#include "../kernels/structs/camera.h"

// Parse file containing <VCGCamera .../> and <ViewSettings .../>
CameraView load_view_state(const std::string& path);

// Parse from in-memory XML string (useful for tests)
CameraView parse_view_state_from_string(const std::string& xml);

std::string dump_view_state_to_string(const CameraView& camera);
