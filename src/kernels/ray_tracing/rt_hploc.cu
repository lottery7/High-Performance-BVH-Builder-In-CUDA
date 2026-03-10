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
    unsigned int nfaces,
    const float* vertices,
    const unsigned int* faces,
    float tMin,
    float& outT,
    int& outFaceId,
    float& outU,
    float& outV)
{
  const int rootIndex = (int)(2u * nfaces - 2u);

  float bestT = FLT_MAX;
  bool hit = false;

  int stack[BVH_STACK_SIZE];
  int sp = 0;
  stack[sp++] = rootIndex;

  while (sp > 0) {
    const int nodeIdx = stack[--sp];
    const BVHNode& node = nodes[nodeIdx];

    float tNear, tFar;
    if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, bestT, tNear, tFar)) continue;

    // leaf
    if (node.left_child_index == INVALID_INDEX) {
      const unsigned int triIdx = node.right_child_index;  // ВАЖНО: как ты описал

      const uint3 face = loadFace(faces, triIdx);
      const float3 v0 = loadVertex(vertices, face.x);
      const float3 v1 = loadVertex(vertices, face.y);
      const float3 v2 = loadVertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, tMin, bestT, false, t, u, v)) {
        hit = true;
        bestT = t;
        outT = t;
        outFaceId = (int)triIdx;
        outU = u;
        outV = v;
      }
      continue;
    }

    // internal: near-first (push far first, then near)
    const int leftIdx = (int)node.left_child_index;
    const int rightIdx = (int)node.right_child_index;

    // если есть шанс на невалидные индексы
    if (leftIdx == (int)INVALID_INDEX || rightIdx == (int)INVALID_INDEX) continue;

    float tNearL, tFarL, tNearR, tFarR;
    const bool hitL = intersect_ray_aabb(orig, dir, nodes[leftIdx].aabb, tMin, bestT, tNearL, tFarL);
    const bool hitR = intersect_ray_aabb(orig, dir, nodes[rightIdx].aabb, tMin, bestT, tNearR, tFarR);

    if (!hitL && !hitR) continue;

    curassert(sp + 2 < BVH_STACK_SIZE, 465130814);

    if (hitL && hitR) {
      if (tNearL <= tNearR) {
        stack[sp++] = rightIdx;  // far
        stack[sp++] = leftIdx;   // near
      } else {
        stack[sp++] = leftIdx;
        stack[sp++] = rightIdx;
      }
    } else if (hitL) {
      stack[sp++] = leftIdx;
    } else {  // hitR
      stack[sp++] = rightIdx;
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
    unsigned int nfaces,
    int ignore_face)
{
  const int rootIndex = (int)(2u * nfaces - 2u);

  const float tMin = 1e-4f;
  float bestT = FLT_MAX;

  int stack[BVH_STACK_SIZE];
  int sp = 0;
  stack[sp++] = rootIndex;

  while (sp > 0) {
    const int nodeIdx = stack[--sp];
    const BVHNode& node = nodes[nodeIdx];

    float tNear, tFar;
    if (!intersect_ray_aabb(orig, dir, node.aabb, tMin, bestT, tNear, tFar)) continue;

    // leaf
    if (node.left_child_index == INVALID_INDEX) {
      const unsigned int triIdx = node.right_child_index;
      if ((int)triIdx == ignore_face) continue;

      const uint3 face = loadFace(faces, triIdx);
      const float3 v0 = loadVertex(vertices, face.x);
      const float3 v1 = loadVertex(vertices, face.y);
      const float3 v2 = loadVertex(vertices, face.z);

      float t, u, v;
      if (intersect_ray_triangle(orig, dir, v0, v1, v2, tMin, bestT, false, t, u, v)) {
        return true;
      }
      continue;
    }

    // internal: near-first
    const int leftIdx = (int)node.left_child_index;
    const int rightIdx = (int)node.right_child_index;
    if (leftIdx == (int)INVALID_INDEX || rightIdx == (int)INVALID_INDEX) continue;

    float tNearL, tFarL, tNearR, tFarR;
    const bool hitL = intersect_ray_aabb(orig, dir, nodes[leftIdx].aabb, tMin, bestT, tNearL, tFarL);
    const bool hitR = intersect_ray_aabb(orig, dir, nodes[rightIdx].aabb, tMin, bestT, tNearR, tFarR);

    if (!hitL && !hitR) continue;

    curassert(sp + 2 < BVH_STACK_SIZE, 136015328);

    if (hitL && hitR) {
      if (tNearL <= tNearR) {
        stack[sp++] = rightIdx;
        stack[sp++] = leftIdx;
      } else {
        stack[sp++] = leftIdx;
        stack[sp++] = rightIdx;
      }
    } else if (hitL) {
      stack[sp++] = leftIdx;
    } else {
      stack[sp++] = rightIdx;
    }
  }

  return false;
}

// + helper: build tangent basis for a given normal
__device__ inline void make_basis(const float3& n, float3& t, float3& b)
{
  // pick a non-parallel vector
  float3 up = (fabsf(n.z) < 0.999f) ? make_float3(0.f, 0.f, 1.f) : make_float3(0.f, 1.f, 0.f);
  t = normalize_f3(cross_f3(up, n));
  b = cross_f3(n, t);
}

__global__ void rt_hploc_kernel(
    const float* vertices,
    const unsigned int* faces,
    const BVHNode* bvh_nodes,
    int* face_id,
    float* ambient_occlusion,
    CameraView* camera,
    unsigned int n_faces)
{
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  curassert(camera->magic_bits_guard == CAMERA_VIEW_MAGIC_BITS_GUARD, 946435342);
  if (i >= camera->K.width || j >= camera->K.height) return;

  float3 ray_origin, ray_direction;
  make_primary_ray(*camera, i + 0.5f, j + 0.5f, ray_origin, ray_direction);

  float tMin = 1e-6f;
  float tBest = FLT_MAX;
  float uBest = 0, vBest = 0;
  int faceIdBest = -1;

  // Use BVH traversal instead of brute-force loop
  closest_hit(ray_origin, ray_direction, bvh_nodes, n_faces, vertices, faces, tMin, tBest, faceIdBest, uBest, vBest);
  ;

  const unsigned int idx = j * camera->K.width + i;
  face_id[idx] = faceIdBest;

  float ao = 1.0f;  // background stays white
  if (faceIdBest >= 0) {
    uint3 f = loadFace(faces, faceIdBest);
    float3 a = loadVertex(vertices, f.x);
    float3 b = loadVertex(vertices, f.y);
    float3 c = loadVertex(vertices, f.z);

    float3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
    float3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
    float3 n = normalize_f3(cross_f3(e1, e2));

    // ensure hemisphere is "outside" relative to the camera ray
    if (n.x * ray_direction.x + n.y * ray_direction.y + n.z * ray_direction.z > 0.0f) n = make_float3(-n.x, -n.y, -n.z);

    float3 P = {ray_origin.x + tBest * ray_direction.x, ray_origin.y + tBest * ray_direction.y, ray_origin.z + tBest * ray_direction.z};

    float scale = fmaxf(fmaxf(length_f3(e1), length_f3(e2)), length_f3(make_float3(c.x - a.x, c.y - a.y, c.z - a.z)));
    float eps = 1e-3f * fmaxf(1.0f, scale);
    float3 Po = {P.x + n.x * eps, P.y + n.y * eps, P.z + n.z * eps};

    // build tangent basis
    float3 T, B;
    make_basis(n, T, B);
    // per-pixel seed (stable)
    union {
      float f32;
      uint32_t u32;
    } tBestUnion;
    tBestUnion.f32 = tBest;
    uint32_t rng = 0x9E3779B9u ^ idx ^ tBestUnion.u32;

    int hits = 0;
    for (int s = 0; s < AO_SAMPLES; ++s) {
      // uniform hemisphere sampling (solid angle)
      float u1 = random01(rng);
      float u2 = random01(rng);
      float z = u1;                     // z in [0,1]
      float phi = 6.28318530718f * u2;  // 2*pi*u2
      float r = sqrtf(fmaxf(0.f, 1.f - z * z));
      float3 d_local = make_float3(r * cosf(phi), r * sinf(phi), z);

      // transform to world space
      float3 d = make_float3(
          T.x * d_local.x + B.x * d_local.y + n.x * d_local.z,
          T.y * d_local.x + B.y * d_local.y + n.y * d_local.z,
          T.z * d_local.x + B.z * d_local.y + n.z * d_local.z);

      if (any_hit_from(Po, d, vertices, faces, bvh_nodes, n_faces, faceIdBest)) ++hits;
    }
    ao = 1.0f - (float)hits / (float)AO_SAMPLES;  // [0,1]
  }
  ambient_occlusion[idx] = ao;
}

namespace cuda
{
  void rt_hploc(
      cudaStream_t stream,
      unsigned int width,
      unsigned int height,
      const float* vertices,
      const unsigned int* faces,
      const BVHNode* bvh_nodes,
      int* face_id,
      float* ambient_occlusion,
      CameraView* camera,
      unsigned int n_faces)
  {
    rt_hploc_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        vertices,
        faces,
        bvh_nodes,
        face_id,
        ambient_occlusion,
        camera,
        n_faces);
  }
}  // namespace cuda
