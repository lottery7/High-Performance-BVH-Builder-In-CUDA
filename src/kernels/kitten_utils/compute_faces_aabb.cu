#include <cuda_runtime.h>

#include "../../utils/utils.h"
#include "KittenEngine/includes/modules/Bound.h"
#include "libbase/runtime_assert.h"

__global__ void compute_faces_aabb_kernel(
    const float* vertices,
    const unsigned int* faces,
    Kitten::Bound<3, float>* face_bounds,
    unsigned int n_faces)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_faces) return;

  // Извлекаем индексы вершин треугольника
  unsigned int i0 = faces[idx * 3 + 0];
  unsigned int i1 = faces[idx * 3 + 1];
  unsigned int i2 = faces[idx * 3 + 2];

  // Создаем AABB для треугольника
  Kitten::Bound<3, float> b(glm::vec3(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]));
  b.absorb(glm::vec3(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]));
  b.absorb(glm::vec3(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]));

  face_bounds[idx] = b;
}

namespace cuda
{
  void compute_faces_aabb(
      cudaStream_t stream,
      const float* d_vertices,
      const unsigned int* d_faces,
      Kitten::Bound<3, float>* d_face_bounds,
      unsigned int n_faces)
  {
    rassert(n_faces > 2, 638420165);
    compute_faces_aabb_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(d_vertices, d_faces, d_face_bounds, n_faces);
  }
}  // namespace cuda
