#include <cfloat>
#include <cstdio>

#include "../../utils/defines.h"
#include "../../utils/utils.h"
#include "../helpers/camera_helpers.cuh"
#include "../helpers/geometry_helpers.cu"
#include "../helpers/random_helpers.cu"
#include "../structs/bvh_node.h"
#include "../structs/camera.h"

constexpr unsigned int bvh_stack_size = 256;

// Эффективная загрузка 80-байтной ноды через 5 128-битных инструкций
__device__ __forceinline__ static BVH8Node load_bvh8_node(const BVH8Node* __restrict__ nodes, unsigned int index)
{
  const uint4* ptr = reinterpret_cast<const uint4*>(&nodes[index]);
  BVH8Node node;
  uint4* node_ptr = reinterpret_cast<uint4*>(&node);
  node_ptr[0] = __ldg(&ptr[0]);
  node_ptr[1] = __ldg(&ptr[1]);
  node_ptr[2] = __ldg(&ptr[2]);
  node_ptr[3] = __ldg(&ptr[3]);
  node_ptr[4] = __ldg(&ptr[4]);
  return node;
}

// Извлечение масштаба сетки: scale = 2^(e - 127). В IEEE754 это эквивалентно сдвигу e << 23
__device__ __forceinline__ float get_grid_scale(std::uint8_t e) { return __uint_as_float(static_cast<std::uint32_t>(e) << 23); }

// Подсчет установленных битов до указанной позиции i
__device__ __forceinline__ std::uint32_t count_bits_below(std::uint32_t mask, std::uint32_t i) { return __popc(mask & ((1u << i) - 1u)); }

__device__ __forceinline__ static float3 inverse_float3(float3 v)
{
  const float eps = 1e-9f;
  if (fabsf(v.x) < eps) v.x = copysignf(eps, v.x);
  if (fabsf(v.y) < eps) v.y = copysignf(eps, v.y);
  if (fabsf(v.z) < eps) v.z = copysignf(eps, v.z);
  return make_float3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
}

// --- PRIMARY RAYS (Поиск ближайшего пересечения) ---

__device__ __forceinline__ static bool closest_hit(
    const unsigned int root_index,
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const BVH8Node* __restrict__ nodes,
    const unsigned int* __restrict__ prim_indices,  // Наш плотный массив геометрии!
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

  // Корень дерева всегда кладем в стек
  stack[sp++] = root_index;

  // Определяем октант луча для аппаратного front-to-back обхода
  std::uint32_t octant = (dir.x < 0.0f ? 4 : 0) | (dir.y < 0.0f ? 2 : 0) | (dir.z < 0.0f ? 1 : 0);

  while (sp > 0) {
    const unsigned int node_index = stack[--sp];
    const BVH8Node node = load_bvh8_node(nodes, node_index);

    // Подготовка "луча" в локальных координатах узла
    float3 o_adj = make_float3((node.p_x - orig.x) * inv_dir.x, (node.p_y - orig.y) * inv_dir.y, (node.p_z - orig.z) * inv_dir.z);

    float3 d_adj = make_float3(get_grid_scale(node.e_x) * inv_dir.x, get_grid_scale(node.e_y) * inv_dir.y, get_grid_scale(node.e_z) * inv_dir.z);

// Идем в обратном порядке (back-to-front).
// Таким образом дальние узлы кладутся в стек первыми, а ближние - последними.
// При извлечении из стека (pop) мы получим строгий front-to-back обход!
#pragma unroll
    for (int order = 7; order >= 0; --order) {
      std::uint32_t i = order ^ (7 - octant);  // Магия из статьи Ylitie et al.

      if (node.meta[i] == 0) continue;  // Слот пустой

      // Сверхбыстрое пересечение луча со сжатым AABB (1 FMA на плоскость)
      float t0x = o_adj.x + node.q_lo_x[i] * d_adj.x;
      float t1x = o_adj.x + node.q_hi_x[i] * d_adj.x;
      float tminx = fminf(t0x, t1x);
      float tmaxx = fmaxf(t0x, t1x);

      float t0y = o_adj.y + node.q_lo_y[i] * d_adj.y;
      float t1y = o_adj.y + node.q_hi_y[i] * d_adj.y;
      float tminy = fmaxf(tminx, fminf(t0y, t1y));
      float tmaxy = fminf(tmaxx, fmaxf(t0y, t1y));

      float t0z = o_adj.z + node.q_lo_z[i] * d_adj.z;
      float t1z = o_adj.z + node.q_hi_z[i] * d_adj.z;
      float tmin = fmaxf(tminy, fminf(t0z, t1z));
      float tmax = fminf(tmaxy, fmaxf(t0z, t1z));

      // Если есть пересечение
      if (tmax >= fmaxf(tmin, t_min) && tmin <= best_t) {
        if ((node.imask & (1u << i)) != 0u) {
          // Это внутренняя нода -> кладем в стек
          stack[sp++] = node.child_base_idx + count_bits_below(node.imask, i);
        } else {
          // Это лист -> проверяем треугольник прямо сейчас
          unsigned int leaf_offset = node.meta[i] & 0x1F;
          unsigned int face_id = prim_indices[node.prim_base_idx + leaf_offset];

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

            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
            out_normal = normalize_f3(cross_f3(e1, e2));
          }
        }
      }
    }
  }

  return hit;
}

// --- AMBIENT OCCLUSION RAYS (Любое пересечение) ---

__device__ static bool any_hit_from(
    const unsigned int root_index,
    const float3& orig,
    const float3& dir,
    const float3& inv_dir,
    const float* __restrict__ vertices,
    const unsigned int* __restrict__ faces,
    const BVH8Node* __restrict__ nodes,
    const unsigned int* __restrict__ prim_indices,
    int ignore_face)
{
  const float t_min = 1e-4f;
  const float t_max = FLT_MAX;

  unsigned int stack[bvh_stack_size];
  unsigned int sp = 0;
  stack[sp++] = root_index;

  while (sp > 0) {
    const unsigned int node_index = stack[--sp];
    const BVH8Node node = load_bvh8_node(nodes, node_index);

    float3 o_adj = make_float3((node.p_x - orig.x) * inv_dir.x, (node.p_y - orig.y) * inv_dir.y, (node.p_z - orig.z) * inv_dir.z);

    float3 d_adj = make_float3(get_grid_scale(node.e_x) * inv_dir.x, get_grid_scale(node.e_y) * inv_dir.y, get_grid_scale(node.e_z) * inv_dir.z);

// Для AO порядок обхода неважен, просто идем линейно 0..7
#pragma unroll
    for (std::uint32_t i = 0; i < 8; ++i) {
      if (node.meta[i] == 0) continue;

      float t0x = o_adj.x + node.q_lo_x[i] * d_adj.x;
      float t1x = o_adj.x + node.q_hi_x[i] * d_adj.x;
      float tminx = fminf(t0x, t1x);
      float tmaxx = fmaxf(t0x, t1x);

      float t0y = o_adj.y + node.q_lo_y[i] * d_adj.y;
      float t1y = o_adj.y + node.q_hi_y[i] * d_adj.y;
      float tminy = fmaxf(tminx, fminf(t0y, t1y));
      float tmaxy = fminf(tmaxx, fmaxf(t0y, t1y));

      float t0z = o_adj.z + node.q_lo_z[i] * d_adj.z;
      float t1z = o_adj.z + node.q_hi_z[i] * d_adj.z;
      float tmin = fmaxf(tminy, fminf(t0z, t1z));
      float tmax = fminf(tmaxy, fmaxf(t0z, t1z));

      if (tmax >= fmaxf(tmin, t_min) && tmin <= t_max) {
        if ((node.imask & (1u << i)) != 0u) {
          stack[sp++] = node.child_base_idx + count_bits_below(node.imask, i);
        } else {
          unsigned int leaf_offset = node.meta[i] & 0x1F;
          unsigned int face_id = prim_indices[node.prim_base_idx + leaf_offset];

          if (static_cast<int>(face_id) == ignore_face) continue;

          const uint3 face = load_face(faces, face_id);
          const float3 v0 = load_vertex(vertices, face.x);
          const float3 v1 = load_vertex(vertices, face.y);
          const float3 v2 = load_vertex(vertices, face.z);

          float t, u, v;
          if (intersect_ray_triangle(orig, dir, v0, v1, v2, t_min, t_max, false, t, u, v)) {
            return true;  // Первое же попадание завершает AO луч
          }
        }
      }
    }
  }

  return false;
}

// --- ГЛАВНОЕ ЯДРО ---

namespace cuda
{
  __global__ void rt_bvh8_kernel(
      const float* vertices,
      const unsigned int* faces,
      const BVH8Node* bvh_nodes,
      const unsigned int* prim_indices,
      const unsigned int root_index,
      int* face_id,
      float* ambient_occlusion,
      const CameraView* camera)
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
        prim_indices,
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
      float3 hit_point =
          make_float3(ray_origin.x + t_best * ray_direction.x, ray_origin.y + t_best * ray_direction.y, ray_origin.z + t_best * ray_direction.z);
      constexpr float eps = 1e-3f;
      float3 offset_origin = make_float3(hit_point.x + n.x * eps, hit_point.y + n.y * eps, hit_point.z + n.z * eps);

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

        if (any_hit_from(root_index, offset_origin, d, inv_d, vertices, faces, bvh_nodes, prim_indices, face_id_best)) {
          ++hits;
        }
      }

      ao = 1.0f - static_cast<float>(hits) / static_cast<float>(AO_SAMPLES);
    }

    ambient_occlusion[idx] = ao;
  }
}  // namespace cuda
