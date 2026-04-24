#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/random_helpers.cu"
#include "../structs//bvh_node.h"
#include "../structs/camera.h"

// closest hit
__device__ bool closest_hit(
    const float3& orig,
    const float3& dir,
    const BVHNode* nodes,
    unsigned int n_faces,
    const float* vertices,
    const unsigned int* faces,
    float t_min,
    float& out_t,
    int& out_face_id,
    float& out_u,
    float& out_v)
{
  const int root_index = static_cast<int>(2u * n_faces - 2u);

  float best_t = FLT_MAX;
  bool hit = false;

  int stack[BVH_STACK_SIZE];
  int sp = 0;
  stack[sp++] = root_index;

  while (sp > 0) {
    const int node_idx = stack[--sp];
    const BVHNode& node = nodes[node_idx];

    float t_near, t_far;
    if (!intersect_ray_aabb(orig, dir, node.aabb, t_min, best_t, t_near, t_far)) continue;

    if (node.is_leaf()) {
      const unsigned int tri_idx = node.right_child_index;

      const uint3 face = loadFace(faces, tri_idx);
      const float3 v0 = loadVertex(vertices, face.x);
      const float3 v1 = loadVertex(vertices, face.y);
      const float3 v2 = loadVertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, best_t, false, t, u, v)) {
        hit = true;
        best_t = t;
        out_t = t;
        out_face_id = static_cast<int>(tri_idx);
        out_u = u;
        out_v = v;
      }
      continue;
    }

    const int left_idx = static_cast<int>(node.left_child_index);
    const int right_idx = static_cast<int>(node.right_child_index);

    if (left_idx == static_cast<int>(INVALID_INDEX) || right_idx == static_cast<int>(INVALID_INDEX)) continue;

    float t_near_l, t_far_l, t_near_r, t_far_r;
    const bool hit_l = intersect_ray_aabb(orig, dir, nodes[left_idx].aabb, t_min, best_t, t_near_l, t_far_l);
    const bool hit_r = intersect_ray_aabb(orig, dir, nodes[right_idx].aabb, t_min, best_t, t_near_r, t_far_r);

    if (!hit_l && !hit_r) continue;

    curassert(sp + 2 < BVH_STACK_SIZE, 465130814);

    if (hit_l && hit_r) {
      if (t_near_l <= t_near_r) {
        stack[sp++] = right_idx;
        stack[sp++] = left_idx;
      } else {
        stack[sp++] = left_idx;
        stack[sp++] = right_idx;
      }
    } else if (hit_l) {
      stack[sp++] = left_idx;
    } else {
      stack[sp++] = right_idx;
    }
  }

  return hit;
}

// any hit (AO)
__device__ bool any_hit_from(
    const float3& orig,
    const float3& dir,
    const float* vertices,
    const unsigned int* faces,
    const BVHNode* nodes,
    unsigned int n_faces,
    int ignore_face)
{
  const int root_index = static_cast<int>(2u * n_faces - 2u);

  const float t_min = 1e-4f;
  float best_t = FLT_MAX;

  int stack[BVH_STACK_SIZE];
  int sp = 0;
  stack[sp++] = root_index;

  while (sp > 0) {
    const int node_idx = stack[--sp];
    const BVHNode& node = nodes[node_idx];

    float t_near, t_far;
    if (!intersect_ray_aabb(orig, dir, node.aabb, t_min, best_t, t_near, t_far)) continue;

    if (node.is_leaf()) {
      const unsigned int tri_idx = node.right_child_index;
      if (static_cast<int>(tri_idx) == ignore_face) continue;

      const uint3 face = loadFace(faces, tri_idx);
      const float3 v0 = loadVertex(vertices, face.x);
      const float3 v1 = loadVertex(vertices, face.y);
      const float3 v2 = loadVertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, best_t, false, t, u, v)) {
        return true;
      }
      continue;
    }

    const int left_idx = static_cast<int>(node.left_child_index);
    const int right_idx = static_cast<int>(node.right_child_index);
    if (left_idx == static_cast<int>(INVALID_INDEX) || right_idx == static_cast<int>(INVALID_INDEX)) continue;

    float t_near_l, t_far_l, t_near_r, t_far_r;
    const bool hit_l = intersect_ray_aabb(orig, dir, nodes[left_idx].aabb, t_min, best_t, t_near_l, t_far_l);
    const bool hit_r = intersect_ray_aabb(orig, dir, nodes[right_idx].aabb, t_min, best_t, t_near_r, t_far_r);

    if (!hit_l && !hit_r) continue;

    curassert(sp + 2 < BVH_STACK_SIZE, 136015328);

    if (hit_l && hit_r) {
      if (t_near_l <= t_near_r) {
        stack[sp++] = right_idx;
        stack[sp++] = left_idx;
      } else {
        stack[sp++] = left_idx;
        stack[sp++] = right_idx;
      }
    } else if (hit_l) {
      stack[sp++] = left_idx;
    } else {
      stack[sp++] = right_idx;
    }
  }

  return false;
}

namespace cuda::hploc
{
  __global__ void rt_hploc_kernel(
      const float* vertices,
      const unsigned int* faces,
      const BVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      const CameraView* camera,
      unsigned int n_faces)
  {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera->magic_bits_guard == CAMERA_VIEW_MAGIC_BITS_GUARD, 946435342);
    if (i >= camera->K.width || j >= camera->K.height) return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float t_best = FLT_MAX;
    float u_best = 0, v_best = 0;
    int face_id_best = -1;

    closest_hit(ray_origin, ray_direction, bvh_nodes, n_faces, vertices, faces, 1e-6f, t_best, face_id_best, u_best, v_best);

    const unsigned int idx = j * camera->K.width + i;
    face_id[idx] = face_id_best;

    float ao = 1.0f;
    if (face_id_best >= 0) {
      uint3 f = loadFace(faces, face_id_best);
      float3 a = loadVertex(vertices, f.x);
      float3 b = loadVertex(vertices, f.y);
      float3 c = loadVertex(vertices, f.z);

      float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
      float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
      float3 n = normalize_f3(cross_f3(e1, e2));

      if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) n = make_float3(-n.x, -n.y, -n.z);

      float3 hit_point = {ray_origin.x + t_best * ray_direction.x, ray_origin.y + t_best * ray_direction.y, ray_origin.z + t_best * ray_direction.z};

      float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)), length_f3(make_float3(c.x - a.x, c.y - a.y, c.z - a.z)));
      float eps = 1e-3f * fmaxf(1.0f, scale);
      float3 offset_origin = {hit_point.x + n.x * eps, hit_point.y + n.y * eps, hit_point.z + n.z * eps};

      float3 tangent, bitangent;
      make_basis(n, tangent, bitangent);
      union {
        float f32;
        uint32_t u32;
      } t_best_union;
      t_best_union.f32 = t_best;
      uint32_t rng = 0x9E3779B9u ^ idx ^ t_best_union.u32;

      int hits = 0;
      for (int s = 0; s < AO_SAMPLES; ++s) {
        float u1 = random01(rng);
        float u2 = random01(rng);
        float z = u1;
        float phi = 6.28318530718f * u2;
        float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);

        float3 d = make_float3(
            tangent.x * d_local.x + bitangent.x * d_local.y + n.x * d_local.z,
            tangent.y * d_local.x + bitangent.y * d_local.y + n.y * d_local.z,
            tangent.z * d_local.x + bitangent.z * d_local.y + n.z * d_local.z);

        if (any_hit_from(offset_origin, d, vertices, faces, bvh_nodes, n_faces, face_id_best)) ++hits;
      }
      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }
    ambient_occlusion[idx] = ao;
  }
}  // namespace cuda::hploc
