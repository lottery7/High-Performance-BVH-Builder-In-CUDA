#include <cfloat>
#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/random_helpers.cu"
#include "rt_hploc_wide.cuh"

constexpr unsigned int hploc_wide_bvh_stack_size = 32;

__device__ __forceinline__ static float3 hploc_wide_inverse_float3(float3 v)
{
  const float eps = 1e-9f;
  if (fabsf(v.x) < eps) v.x = copysignf(eps, v.x);
  if (fabsf(v.y) < eps) v.y = copysignf(eps, v.y);
  if (fabsf(v.z) < eps) v.z = copysignf(eps, v.z);
  return make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

__device__ __forceinline__ static float hploc_wide_intersect_ray_aabb_inv(
    const float3& ray_o,
    const float3& ray_inv_d,
    const AABB& box,
    float t_min,
    float t_max)
{
  const float t0_x = (box.min_x - ray_o.x) * ray_inv_d.x;
  const float t1_x = (box.max_x - ray_o.x) * ray_inv_d.x;
  float tmin = fminf(t0_x, t1_x);
  float tmax = fmaxf(t0_x, t1_x);

  const float t0_y = (box.min_y - ray_o.y) * ray_inv_d.y;
  const float t1_y = (box.max_y - ray_o.y) * ray_inv_d.y;
  tmin = fmaxf(tmin, fminf(t0_y, t1_y));
  tmax = fminf(tmax, fmaxf(t0_y, t1_y));

  const float t0_z = (box.min_z - ray_o.z) * ray_inv_d.z;
  const float t1_z = (box.max_z - ray_o.z) * ray_inv_d.z;
  tmin = fmaxf(tmin, fminf(t0_z, t1_z));
  tmax = fminf(tmax, fmaxf(t0_z, t1_z));

  tmin = fmaxf(tmin, t_min);
  tmax = fminf(tmax, t_max);

  return (tmax >= tmin) ? tmin : -1.0f;
}

template <unsigned int Arity>
__device__ __forceinline__ static WideBVHNode<Arity> hploc_wide_load_node(const WideBVHNode<Arity>* __restrict__ nodes, unsigned int index)
{
  return nodes[index];
}

template <unsigned int Arity>
__device__ __forceinline__ static bool hploc_wide_closest_hit(
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const WideBVHNode<Arity>* __restrict__ nodes,
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ faces,
    float t_min,
    float& out_t,
    int& out_face_id,
    float3& out_normal)
{
  float best_t = FLT_MAX;
  bool hit = false;

  unsigned int stack[hploc_wide_bvh_stack_size];
  unsigned int sp = 0;
  unsigned int node_index = 0;

  while (true) {
    const WideBVHNode<Arity> node = hploc_wide_load_node(nodes, node_index);
    curassert(!node.is_leaf(), 245830114);

    unsigned int hit_indices[Arity];
    float hit_t_near[Arity];
    unsigned int hit_count = 0;

    for (unsigned int slot = 0; slot < Arity; ++slot) {
      const unsigned int child_index = node.children[slot];
      if (child_index == INVALID_INDEX) continue;

      const WideBVHNode<Arity> child = hploc_wide_load_node(nodes, child_index);

      const float t_near = hploc_wide_intersect_ray_aabb_inv(orig, inv_dir, child.aabb, t_min, best_t);
      if (t_near < 0.0f) continue;

      unsigned int insert_pos = hit_count;
      while (insert_pos > 0 && t_near < hit_t_near[insert_pos - 1]) {
        hit_t_near[insert_pos] = hit_t_near[insert_pos - 1];
        hit_indices[insert_pos] = hit_indices[insert_pos - 1];
        --insert_pos;
      }
      hit_t_near[insert_pos] = t_near;
      hit_indices[insert_pos] = child_index;
      ++hit_count;
    }

    unsigned int internal_indices[Arity];
    unsigned int internal_count = 0;

    for (unsigned int hit_i = 0; hit_i < hit_count; ++hit_i) {
      if (hit_t_near[hit_i] > best_t) continue;

      const unsigned int child_index = hit_indices[hit_i];
      const WideBVHNode<Arity> child = hploc_wide_load_node(nodes, child_index);

      if (child.is_leaf()) {
        const unsigned int face_id = child.children[1];
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

          const float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
          const float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
          out_normal = normalize_f3(cross_f3(e1, e2));
        }
      } else {
        internal_indices[internal_count++] = child_index;
      }
    }

    if (internal_count > 0) {
      for (unsigned int i = internal_count - 1; i > 0; --i) {
        curassert(sp < hploc_wide_bvh_stack_size, 245830113);
        stack[sp++] = internal_indices[i];
      }
      node_index = internal_indices[0];
    } else {
      if (sp == 0) break;
      node_index = stack[--sp];
    }
  }

  return hit;
}

template <unsigned int Arity>
__device__ static bool hploc_wide_any_hit_from(
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ faces,
    const WideBVHNode<Arity>* __restrict__ nodes,
    int ignore_face,
    const float t_max)
{
  const float t_min = 1e-4f;

  unsigned int stack[hploc_wide_bvh_stack_size];
  unsigned int sp = 0;
  unsigned int node_index = 0;

  while (true) {
    const WideBVHNode<Arity> node = hploc_wide_load_node(nodes, node_index);
    curassert(!node.is_leaf(), 420833046);

    unsigned int next_node_index = 0;
    bool has_next_node = false;

    for (unsigned int slot = 0; slot < Arity; ++slot) {
      const unsigned int child_index = node.children[slot];
      if (child_index == INVALID_INDEX) continue;

      const WideBVHNode<Arity> child = hploc_wide_load_node(nodes, child_index);

      const float t_near = hploc_wide_intersect_ray_aabb_inv(orig, inv_dir, child.aabb, t_min, t_max);
      if (t_near < 0.0f) continue;

      if (child.is_leaf()) {
        const unsigned int face_id = child.children[1];
        if (static_cast<int>(face_id) == ignore_face) continue;

        const uint3 face = load_face(faces, face_id);
        const float3 v0 = load_vertex(vertices, face.x);
        const float3 v1 = load_vertex(vertices, face.y);
        const float3 v2 = load_vertex(vertices, face.z);

        float t, u, v;
        if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, t_max, false, t, u, v)) return true;
      } else {
        if (has_next_node) {
          curassert(sp < hploc_wide_bvh_stack_size, 420833045);
          stack[sp++] = next_node_index;
        }
        next_node_index = child_index;
        has_next_node = true;
      }
    }

    if (has_next_node) {
      node_index = next_node_index;
    } else {
      if (sp == 0) break;
      node_index = stack[--sp];
    }
  }

  return false;
}

namespace cuda
{
  template <unsigned int Arity>
  __global__ void rt_hploc_wide_kernel(
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode<Arity>* bvh_nodes,
      float* ambient_occlusion,
      const float* ao_radius,
      const CameraView* camera)
  {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera->magic_bits_guard == CAMERA_VIEW_MAGIC_BITS_GUARD, 144090086);
    if (i >= camera->K.width || j >= camera->K.height) return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);
    const float3 inv_ray_direction = hploc_wide_inverse_float3(ray_direction);

    float t_best = FLT_MAX;
    int face_id_best = -1;
    float3 face_normal;

    hploc_wide_closest_hit(ray_origin, ray_direction, inv_ray_direction, bvh_nodes, vertices, faces, 1e-6f, t_best, face_id_best, face_normal);

    const unsigned int idx = j * camera->K.width + i;
    const float ao_radius_value = *ao_radius;

    float ao = 1.0f;
    if (face_id_best >= 0) {
      float3 n = face_normal;
      if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) n = make_float3(-n.x, -n.y, -n.z);

      const float3 hit_point =
          make_float3(ray_origin.x + t_best * ray_direction.x, ray_origin.y + t_best * ray_direction.y, ray_origin.z + t_best * ray_direction.z);
      constexpr float eps = 1e-3f;
      const float3 offset_origin = make_float3(hit_point.x + n.x * eps, hit_point.y + n.y * eps, hit_point.z + n.z * eps);

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
        const float u1 = random01(rng);
        const float u2 = random01(rng);
        const float z = u1;
        const float phi = 6.28318530718f * u2;
        const float r = sqrtf(fmaxf(0.f, 1.f - z * z));
        const float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);
        const float3 d = make_float3(
            tangent.x * d_local.x + bitangent.x * d_local.y + n.x * d_local.z,
            tangent.y * d_local.x + bitangent.y * d_local.y + n.y * d_local.z,
            tangent.z * d_local.x + bitangent.z * d_local.y + n.z * d_local.z);
        const float3 inv_d = hploc_wide_inverse_float3(d);

        if (hploc_wide_any_hit_from(offset_origin, d, inv_d, vertices, faces, bvh_nodes, face_id_best, ao_radius_value)) ++hits;
      }
      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }

    ambient_occlusion[idx] = ao;
  }

  template __global__ void rt_hploc_wide_kernel<4>(
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode4* bvh_nodes,
      float* ambient_occlusion,
      const float* ao_radius,
      const CameraView* camera);

  template __global__ void rt_hploc_wide_kernel<8>(
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode8* bvh_nodes,
      float* ambient_occlusion,
      const float* ao_radius,
      const CameraView* camera);
}  // namespace cuda
