#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/random_helpers.cu"
#include "../structs//bvh_node.h"
#include "../structs/camera.h"

constexpr unsigned int bvh_stack_size = 32;

__device__ __forceinline__ static BVH2Node load_bvh_node(const BVH2Node* __restrict__ nodes, unsigned int index)
{
  const uint4* ptr = reinterpret_cast<const uint4*>(&nodes[index]);
  uint4 v0 = __ldg(&ptr[0]);
  uint4 v1 = __ldg(&ptr[1]);
  BVH2Node node;
  uint4* node_ptr = reinterpret_cast<uint4*>(&node);
  node_ptr[0] = v0;
  node_ptr[1] = v1;
  return node;
}

__device__ __forceinline__ static float3 inverse_float3(float3 v)
{
  const float eps = 1e-9f;
  if (fabsf(v.x) < eps) v.x = copysignf(eps, v.x);
  if (fabsf(v.y) < eps) v.y = copysignf(eps, v.y);
  if (fabsf(v.z) < eps) v.z = copysignf(eps, v.z);
  return make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

__device__ __forceinline__ static float intersect_ray_aabb_inv(const float3& ray_o, const float3& ray_inv_d, const AABB& box, float tMin, float tMax)
{
  float t0_x = (box.min_x - ray_o.x) * ray_inv_d.x;
  float t1_x = (box.max_x - ray_o.x) * ray_inv_d.x;
  float tmin = fminf(t0_x, t1_x);
  float tmax = fmaxf(t0_x, t1_x);

  float t0_y = (box.min_y - ray_o.y) * ray_inv_d.y;
  float t1_y = (box.max_y - ray_o.y) * ray_inv_d.y;
  tmin = fmaxf(tmin, fminf(t0_y, t1_y));
  tmax = fminf(tmax, fmaxf(t0_y, t1_y));

  float t0_z = (box.min_z - ray_o.z) * ray_inv_d.z;
  float t1_z = (box.max_z - ray_o.z) * ray_inv_d.z;
  tmin = fmaxf(tmin, fminf(t0_z, t1_z));
  tmax = fminf(tmax, fmaxf(t0_z, t1_z));

  tmin = fmaxf(tmin, tMin);
  tmax = fminf(tmax, tMax);

  return (tmax >= tmin) ? tmin : -1.0f;
}

__device__ __forceinline__ static bool closest_hit(
    const unsigned int root_index,
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const BVH2Node* __restrict__ nodes,
    unsigned int n_faces,
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ faces,
    float t_min,
    float& out_t,
    int& out_face_id,
    float3& out_normal)
{
  float best_t = FLT_MAX;
  bool hit = false;

  unsigned int stack[bvh_stack_size];
  unsigned int sp = 0;

  if (intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, root_index).aabb, t_min, best_t) >= 0.0f) {
    stack[sp++] = root_index;
  }

  while (sp > 0) {
    const unsigned int node_index = stack[--sp];
    // TODO расставить ldg на чтении нод
    BVH2Node node = load_bvh_node(nodes, node_index);

    if (node.is_leaf()) {
      const unsigned int face_id = node.right_child_index;
      const uint3 face = load_face(faces, face_id);
      const float3 v0 = load_vertex(vertices, face.x);
      const float3 v1 = load_vertex(vertices, face.y);
      const float3 v2 = load_vertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, best_t, false, t, u, v)) {
        hit = true;
        best_t = t;
        out_t = t;
        out_face_id = static_cast<int>(face_id);

        float3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        float3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        out_normal = normalize_f3(cross_f3(e1, e2));
      }
      continue;
    }

    const unsigned int left_child = node.left_child_index;
    const unsigned int right_child = node.right_child_index;

    float t_l = intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, left_child).aabb, t_min, best_t);
    float t_r = intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, right_child).aabb, t_min, best_t);

    if (t_l >= 0.0f && t_r >= 0.0f) {
      if (t_l > t_r) {
        stack[sp++] = left_child;
        stack[sp++] = right_child;
      } else {
        stack[sp++] = right_child;
        stack[sp++] = left_child;
      }
    } else if (t_l >= 0.0f) {
      stack[sp++] = left_child;
    } else if (t_r >= 0.0f) {
      stack[sp++] = right_child;
    }
  }

  return hit;
}

__device__ static bool any_hit_from(
    const unsigned int root_index,
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ faces,
    const BVH2Node* __restrict__ nodes,
    unsigned int n_faces,
    int ignore_face)
{
  const float t_min = 1e-4f;
  const float t_max = FLT_MAX;

  unsigned int stack[bvh_stack_size];
  unsigned int sp = 0;

  if (intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, root_index).aabb, t_min, t_max) >= 0.0f) {
    stack[sp++] = root_index;
  }

  while (sp > 0) {
    const unsigned int node_index = stack[--sp];
    const BVH2Node& node = load_bvh_node(nodes, node_index);

    if (node.is_leaf()) {
      const unsigned int face_id = node.right_child_index;

      if (static_cast<int>(face_id) == ignore_face) continue;

      const uint3 face = load_face(faces, face_id);
      const float3 v0 = load_vertex(vertices, face.x);
      const float3 v1 = load_vertex(vertices, face.y);
      const float3 v2 = load_vertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, t_max, false, t, u, v)) {
        return true;
      }
      continue;
    }

    const unsigned int left_child = node.left_child_index;
    const unsigned int right_child = node.right_child_index;

    float t_l = intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, left_child).aabb, t_min, t_max);
    float t_r = intersect_ray_aabb_inv(orig, inv_dir, load_bvh_node(nodes, right_child).aabb, t_min, t_max);

    if (t_r >= 0.0f) stack[sp++] = right_child;
    if (t_l >= 0.0f) stack[sp++] = left_child;
  }

  return false;
}

namespace cuda
{
  __global__ void rt_bvh2_kernel(
      const float* vertices,
      const unsigned int* faces,
      const BVH2Node* bvh_nodes,
      const unsigned int root_index,
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
    float3 inv_ray_direction = inverse_float3(ray_direction);

    float t_best = FLT_MAX;
    int face_id_best = -1;

    float3 face_normal;
    closest_hit(
        root_index,
        ray_origin,
        ray_direction,
        inv_ray_direction,
        bvh_nodes,
        n_faces,
        vertices,
        faces,
        1e-6f,
        t_best,
        face_id_best,
        face_normal);

    const unsigned int idx = j * camera->K.width + i;
    face_id[idx] = face_id_best;

    float ao = 1.0f;

    if (face_id_best >= 0) {
      float3 n = face_normal;
      if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) n = make_float3(-n.x, -n.y, -n.z);
      float3 hit_point = {ray_origin.x + t_best * ray_direction.x, ray_origin.y + t_best * ray_direction.y, ray_origin.z + t_best * ray_direction.z};
      constexpr float eps = 1e-3f;
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

#pragma unroll
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
        float3 inv_d = inverse_float3(d);

        if (any_hit_from(root_index, offset_origin, d, inv_d, vertices, faces, bvh_nodes, n_faces, face_id_best)) ++hits;
      }

      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }

    ambient_occlusion[idx] = ao;
  }
}  // namespace cuda
