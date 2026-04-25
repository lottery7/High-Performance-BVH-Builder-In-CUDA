#pragma once

#include <string>
#include <vector>

#include "../kernels/structs/aabb.h"
#include "libbase/point.h"

// Minimal scene container: triangles by vertex indices + positions
struct SceneGeometry {
  std::vector<point3f> vertices;  // each vertex is (x, y, z)
  std::vector<point3u> faces;     // each face is (v0, v1, v2) - indices of three vertices used in a triangle
};

// Loads geometry from Wavefront OBJ (supports v, f; polygons are fan-triangulated)
SceneGeometry load_obj(const std::string &path);

// Loads geometry from PLY (ascii or binary_little_endian 1.0; x/y/z + face index list)
SceneGeometry load_ply(const std::string &path);

// Dispatch by file extension (.ply / .obj)
SceneGeometry load_scene(const std::string &path);
