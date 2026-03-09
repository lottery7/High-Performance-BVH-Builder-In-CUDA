#pragma once

#include <cuda_runtime_api.h>

#include "../../io/scene_reader.h"
#include "../../kernels/kernels.h"
#include "../../utils/cuda_utils.h"
#include "camera.h"

namespace cuda
{
  struct Scene {
    cudaStream_t stream;

    float* d_vertices = nullptr;
    unsigned int n_vertices;

    unsigned int* d_faces = nullptr;
    unsigned int n_faces;

    CameraView* d_camera = nullptr;
    AABB aabb;

    Scene(cudaStream_t stream, const SceneGeometry& scene, const CameraView& cam)
        : stream(stream), n_vertices(scene.vertices.size()), n_faces(scene.faces.size()), aabb(scene.aabb)
    {
      static_assert(sizeof(point3f) == 12);
      static_assert(sizeof(point3u) == 12);

      CUDA_SAFE_CALL(cudaMallocAsync(&d_vertices, 3 * n_vertices * sizeof(float), stream));
      CUDA_SAFE_CALL(cudaMallocAsync(&d_faces, 3 * n_faces * sizeof(unsigned int), stream));
      CUDA_SAFE_CALL(cudaMallocAsync(&d_camera, sizeof(CameraView), stream));

      CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertices, scene.vertices.data(), 3 * n_vertices * sizeof(float), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_faces, scene.faces.data(), 3 * n_faces * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_camera, &cam, sizeof(CameraView), cudaMemcpyHostToDevice, stream));
    }

    ~Scene()
    {
      cudaFreeAsync(d_vertices, stream);
      cudaFreeAsync(d_faces, stream);
      cudaFreeAsync(d_camera, stream);
    }
  };
}  // namespace cuda
