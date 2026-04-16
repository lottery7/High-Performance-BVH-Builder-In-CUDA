#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/random_helpers.cu"
#include "../structs/camera.h"
#include "../structs/wide_bvh_node.h"

namespace
{
  __device__ __forceinline__ bool is_internal_child(const WideBVHNode& node, unsigned int slot) { return ((node.internal_mask >> slot) & 1u) != 0; }

  __device__ __forceinline__ bool is_valid_child(const WideBVHNode& node, unsigned int slot) { return ((node.valid_mask >> slot) & 1u) != 0; }

  __device__ __forceinline__ void test_leaf_triangle(
      const float3& orig,
      const float3& dir,
      const float* vertices,
      const unsigned int* faces,
      float t_min,
      float t_max,
      unsigned int tri_idx,
      bool& hit,
      float& out_t,
      int& out_face_id,
      float& out_u,
      float& out_v)
  {
    const uint3 face = loadFace(faces, tri_idx);
    const float3 v0 = loadVertex(vertices, face.x);
    const float3 v1 = loadVertex(vertices, face.y);
    const float3 v2 = loadVertex(vertices, face.z);

    float t, u, v;
    if (!intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, t_max, false, t, u, v)) return;

    hit = true;
    out_t = t;
    out_face_id = static_cast<int>(tri_idx);
    out_u = u;
    out_v = v;
  }

  __device__ bool wide_closest_hit(
      const float3& orig,
      const float3& dir,
      const WideBVHNode* nodes,
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

    unsigned int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0u;

    while (sp > 0) {
      const unsigned int node_index = stack[--sp];
      const WideBVHNode& node = nodes[node_index];

      float child_t_near[8];
      unsigned int child_node_indices[8];
      int child_count = 0;

      for (unsigned int slot = 0; slot < 8; ++slot) {
        if (!is_valid_child(node, slot)) continue;

        float t_near, t_far;
        if (!intersect_ray_aabb(orig, dir, node.child_aabbs[slot], t_min, best_t, t_near, t_far)) continue;

        if (is_internal_child(node, slot)) {
          int insert_pos = child_count;
          while (insert_pos > 0 && child_t_near[insert_pos - 1] > t_near) {
            child_t_near[insert_pos] = child_t_near[insert_pos - 1];
            child_node_indices[insert_pos] = child_node_indices[insert_pos - 1];
            --insert_pos;
          }
          child_t_near[insert_pos] = t_near;
          child_node_indices[insert_pos] = node.child_indices[slot];
          ++child_count;
        } else {
          test_leaf_triangle(orig, dir, vertices, faces, t_min, best_t, node.child_indices[slot], hit, best_t, out_face_id, out_u, out_v);
          if (hit) out_t = best_t;
        }
      }

      curassert(sp + child_count < BVH_STACK_SIZE, 728384541);
      for (int i = child_count - 1; i >= 0; --i) {
        stack[sp++] = child_node_indices[i];
      }
    }

    return hit;
  }

  __device__ bool wide_any_hit_from(
      const float3& orig,
      const float3& dir,
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode* nodes,
      int ignore_face)
  {
    const float t_min = 1e-4f;
    float best_t = FLT_MAX;

    unsigned int stack[BVH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0u;

    while (sp > 0) {
      const unsigned int node_index = stack[--sp];
      const WideBVHNode& node = nodes[node_index];

      float child_t_near[8];
      unsigned int child_node_indices[8];
      int child_count = 0;

      for (unsigned int slot = 0; slot < 8; ++slot) {
        if (!is_valid_child(node, slot)) continue;

        float t_near, t_far;
        if (!intersect_ray_aabb(orig, dir, node.child_aabbs[slot], t_min, best_t, t_near, t_far)) continue;

        if (is_internal_child(node, slot)) {
          int insert_pos = child_count;
          while (insert_pos > 0 && child_t_near[insert_pos - 1] > t_near) {
            child_t_near[insert_pos] = child_t_near[insert_pos - 1];
            child_node_indices[insert_pos] = child_node_indices[insert_pos - 1];
            --insert_pos;
          }
          child_t_near[insert_pos] = t_near;
          child_node_indices[insert_pos] = node.child_indices[slot];
          ++child_count;
          continue;
        }

        const unsigned int tri_idx = node.child_indices[slot];
        if (static_cast<int>(tri_idx) == ignore_face) continue;

        const uint3 face = loadFace(faces, tri_idx);
        const float3 v0 = loadVertex(vertices, face.x);
        const float3 v1 = loadVertex(vertices, face.y);
        const float3 v2 = loadVertex(vertices, face.z);

        float t, u, v;
        if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, best_t, false, t, u, v)) return true;
      }

      curassert(sp + child_count < BVH_STACK_SIZE, 621805413);
      for (int i = child_count - 1; i >= 0; --i) {
        stack[sp++] = child_node_indices[i];
      }
    }

    return false;
  }

  __device__ inline void make_basis(const float3& n, float3& t, float3& b)
  {
    const float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f) : make_float3(0.f, 1.f, 0.f);
    t = normalize_f3(cross_f3(up, n));
    b = cross_f3(n, t);
  }

  __global__ void rt_hploc_wide_kernel(
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera)
  {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    curassert(camera->magic_bits_guard == CAMERA_VIEW_MAGIC_BITS_GUARD, 946435342);
    if (i >= camera->K.width || j >= camera->K.height) return;

    float3 ray_origin, ray_direction;
    make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

    float t_min = 1e-6f;
    float t_best = FLT_MAX;
    float u_best = 0.0f;
    float v_best = 0.0f;
    int face_id_best = -1;

    wide_closest_hit(ray_origin, ray_direction, bvh_nodes, vertices, faces, t_min, t_best, face_id_best, u_best, v_best);

    const unsigned int idx = j * camera->K.width + i;
    face_id[idx] = face_id_best;

    float ao = 1.0f;
    if (face_id_best >= 0) {
      const uint3 f = loadFace(faces, face_id_best);
      const float3 a = loadVertex(vertices, f.x);
      const float3 b = loadVertex(vertices, f.y);
      const float3 c = loadVertex(vertices, f.z);

      const float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
      const float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
      float3 n = normalize_f3(cross_f3(e1, e2));

      if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) n = make_float3(-n.x, -n.y, -n.z);

      const float3 p = {ray_origin.x + t_best * ray_direction.x, ray_origin.y + t_best * ray_direction.y, ray_origin.z + t_best * ray_direction.z};

      const float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)), length_f3(make_float3(c.x - a.x, c.y - a.y, c.z - a.z)));
      const float eps = 1e-3f * fmaxf(1.0f, scale);
      const float3 po = {p.x + n.x * eps, p.y + n.y * eps, p.z + n.z * eps};

      float3 t, basis_b;
      make_basis(n, t, basis_b);

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
            t.x * d_local.x + basis_b.x * d_local.y + n.x * d_local.z,
            t.y * d_local.x + basis_b.y * d_local.y + n.y * d_local.z,
            t.z * d_local.x + basis_b.z * d_local.y + n.z * d_local.z);

        if (wide_any_hit_from(po, d, vertices, faces, bvh_nodes, face_id_best)) ++hits;
      }
      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }

    ambient_occlusion[idx] = ao;
  }
}  // namespace

namespace cuda
{
  void rt_hploc_wide(
      cudaStream_t stream,
      unsigned int width,
      unsigned int height,
      const float* vertices,
      const unsigned int* faces,
      const WideBVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera)
  {
    rt_hploc_wide_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(vertices, faces, bvh_nodes, face_id, ambient_occlusion, camera);
  }
}  // namespace cuda
