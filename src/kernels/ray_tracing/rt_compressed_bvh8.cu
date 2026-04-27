#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/helpers.cuh"
#include "../helpers/random_helpers.cu"
#include "../structs/camera.h"
#include "kernels/nexus_bvh/nexus_bvh8.cuh"

using CompressedBVH8Node = cuda::nexus_bvh_wide::BVH8Node;
using BVH8NodeExplicit = cuda::nexus_bvh_wide::BVH8NodeExplicit;

__device__ __forceinline__ unsigned int count_bits_below(unsigned int x, unsigned int i)
{
  const unsigned int mask = (1u << i) - 1u;
  return __popc(x & mask);
}

__device__ __forceinline__ AABB decode_child_aabb(const BVH8NodeExplicit& node, unsigned int slot)
{
  const float ex = __uint_as_float(static_cast<unsigned int>(node.e[0]) << 23);
  const float ey = __uint_as_float(static_cast<unsigned int>(node.e[1]) << 23);
  const float ez = __uint_as_float(static_cast<unsigned int>(node.e[2]) << 23);

  AABB child_aabb;
  child_aabb.min_x = node.p.x + ex * node.qlox[slot];
  child_aabb.min_y = node.p.y + ey * node.qloy[slot];
  child_aabb.min_z = node.p.z + ez * node.qloz[slot];
  child_aabb.max_x = node.p.x + ex * node.qhix[slot];
  child_aabb.max_y = node.p.y + ey * node.qhiy[slot];
  child_aabb.max_z = node.p.z + ez * node.qhiz[slot];
  return child_aabb;
}

__device__ __forceinline__ unsigned int decode_internal_child_index(const BVH8NodeExplicit& node, unsigned int slot)
{
  return node.childBaseIdx + count_bits_below(node.imask, slot);
}

__device__ __forceinline__ unsigned int decode_leaf_primitive_index(const BVH8NodeExplicit& node, unsigned int slot, const unsigned int* prim_idx)
{
  const unsigned int leaf_offset = node.meta[slot] & 0x1fu;
  return prim_idx[node.primBaseIdx + leaf_offset];
}

static __device__ bool hploc_wide8_closest_hit(
    const float3& orig,
    const float3& dir,
    const CompressedBVH8Node* nodes,
    const unsigned int* prim_idx,
    const float* vertices,
    const unsigned int* faces,
    float t_min,
    float& out_t,
    int& out_face_id,
    float& out_u,
    float& out_v)
{
  float best_t = FLT_MAX;
  bool hit = false;

  constexpr int stack_capacity = BVH_STACK_SIZE * 8;
  int stack[stack_capacity];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    const int node_idx = stack[--sp];
    const auto& node = reinterpret_cast<const BVH8NodeExplicit&>(nodes[node_idx]);

    unsigned int hit_indices[8];
    float hit_t_near[8];
    unsigned int hit_count = 0;

    for (unsigned int slot = 0; slot < 8; ++slot) {
      if (node.meta[slot] == 0) continue;

      const AABB child_aabb = decode_child_aabb(node, slot);

      float t_near, t_far;
      if (!intersect_ray_aabb(orig, dir, child_aabb, t_min, best_t, t_near, t_far)) continue;

      const bool internal = (node.imask & (1u << slot)) != 0u;
      if (!internal) {
        const unsigned int tri_idx = decode_leaf_primitive_index(node, slot, prim_idx);
        const uint3 face = load_face(faces, tri_idx);
        const float3 v0 = load_vertex(vertices, face.x);
        const float3 v1 = load_vertex(vertices, face.y);
        const float3 v2 = load_vertex(vertices, face.z);

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

      unsigned int insert_pos = hit_count;
      while (insert_pos > 0 && t_near < hit_t_near[insert_pos - 1]) {
        hit_t_near[insert_pos] = hit_t_near[insert_pos - 1];
        hit_indices[insert_pos] = hit_indices[insert_pos - 1];
        --insert_pos;
      }
      hit_t_near[insert_pos] = t_near;
      hit_indices[insert_pos] = decode_internal_child_index(node, slot);
      ++hit_count;
    }

    curassert(sp + static_cast<int>(hit_count) < stack_capacity, 23295145);
    for (int i = static_cast<int>(hit_count) - 1; i >= 0; --i) {
      stack[sp++] = static_cast<int>(hit_indices[i]);
    }
  }

  return hit;
}

static __device__ bool hploc_wide8_any_hit_from(
    const float3& orig,
    const float3& dir,
    const float* vertices,
    const unsigned int* faces,
    const CompressedBVH8Node* nodes,
    const unsigned int* prim_idx,
    int ignore_face)
{
  constexpr int stack_capacity = BVH_STACK_SIZE * 8;
  int stack[stack_capacity];
  int sp = 0;
  stack[sp++] = 0;

  const float t_min = 1e-4f;
  float best_t = FLT_MAX;

  while (sp > 0) {
    const int node_idx = stack[--sp];
    const auto& node = reinterpret_cast<const BVH8NodeExplicit&>(nodes[node_idx]);

    for (unsigned int slot = 0; slot < 8; ++slot) {
      if (node.meta[slot] == 0) continue;

      const AABB child_aabb = decode_child_aabb(node, slot);

      float t_near, t_far;
      if (!intersect_ray_aabb(orig, dir, child_aabb, t_min, best_t, t_near, t_far)) continue;

      const bool internal = (node.imask & (1u << slot)) != 0u;
      if (!internal) {
        const unsigned int tri_idx = decode_leaf_primitive_index(node, slot, prim_idx);
        if (static_cast<int>(tri_idx) == ignore_face) continue;

        const uint3 face = load_face(faces, tri_idx);
        const float3 v0 = load_vertex(vertices, face.x);
        const float3 v1 = load_vertex(vertices, face.y);
        const float3 v2 = load_vertex(vertices, face.z);

        float t, u, v;
        if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, best_t, false, t, u, v)) return true;
        continue;
      }

      curassert(sp < stack_capacity, 420833046);
      stack[sp++] = static_cast<int>(decode_internal_child_index(node, slot));
    }
  }

  return false;
}

namespace cuda
{
  __global__ void rt_compressed_bvh8_kernel(
      const float* vertices,
      const unsigned int* faces,
      const nexus_bvh_wide::BVH8Node* bvh_nodes,
      const unsigned int* prim_idx,
      int* face_id,
      float* ambient_occlusion,
      const CameraView* camera)
  {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera->magic_bits_guard == CAMERA_VIEW_MAGIC_BITS_GUARD, 144090087);
    if (i >= camera->K.width || j >= camera->K.height) return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float t_best = FLT_MAX;
    float u_best = 0.0f;
    float v_best = 0.0f;
    int face_id_best = -1;

    hploc_wide8_closest_hit(ray_origin, ray_direction, bvh_nodes, prim_idx, vertices, faces, 1e-6f, t_best, face_id_best, u_best, v_best);

    const unsigned int idx = j * camera->K.width + i;
    face_id[idx] = face_id_best;

    float ao = 1.0f;
    if (face_id_best >= 0) {
      uint3 f = load_face(faces, face_id_best);
      float3 a = load_vertex(vertices, f.x);
      float3 b = load_vertex(vertices, f.y);
      float3 c = load_vertex(vertices, f.z);

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

        if (hploc_wide8_any_hit_from(offset_origin, d, vertices, faces, bvh_nodes, prim_idx, face_id_best)) ++hits;
      }
      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }
    ambient_occlusion[idx] = ao;
  }
}  // namespace cuda
