#pragma once

#include <cuda_runtime_api.h>
#include <libimages/debug_io.h>

#include "../io/scene_reader.h"
#include "../kernels/defines.h"
#include "../kernels/kernels.h"
#include "../utils/cuda_utils.h"

class SceneDevice
{
  cudaStream_t stream_;

 public:
  float* d_vertices = nullptr;
  unsigned int* d_faces = nullptr;
  CameraView* d_camera = nullptr;

  unsigned int n_vertices = 0;
  unsigned int n_faces = 0;
  AABB aabb;

  SceneDevice(cudaStream_t stream, const SceneGeometry& scene, const CameraView& cam)
      : stream_(stream), n_vertices(scene.vertices.size()), n_faces(scene.faces.size()), aabb(scene.aabb)
  {
    static_assert(sizeof(point3f) == 12);
    static_assert(sizeof(point3u) == 12);

    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_vertices), 3 * n_vertices * sizeof(float), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_faces), 3 * n_faces * sizeof(unsigned int), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_camera), sizeof(CameraView), stream_));

    CUDA_SAFE_CALL(cudaMemcpyAsync(d_vertices, scene.vertices.data(), 3 * n_vertices * sizeof(float), cudaMemcpyHostToDevice, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_faces, scene.faces.data(), 3 * n_faces * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_camera, &cam, sizeof(CameraView), cudaMemcpyHostToDevice, stream_));

    CUDA_CHECK_STREAM(stream_);
  }

  ~SceneDevice()
  {
    cudaFreeAsync(d_vertices, stream_);
    cudaFreeAsync(d_faces, stream_);
    cudaFreeAsync(d_camera, stream_);
    cudaStreamSynchronize(stream_);
  }
};

class FramebuffersDevice
{
  cudaStream_t stream_;

 public:
  int* d_face_id = nullptr;
  float* d_ao = nullptr;

  unsigned int width = 0;
  unsigned int height = 0;

  FramebuffersDevice(cudaStream_t stream, unsigned int w, unsigned int h) : stream_(stream), width(w), height(h)
  {
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_face_id), w * h * sizeof(int), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&d_ao), w * h * sizeof(float), stream_));
    CUDA_CHECK_STREAM(stream_);
  }

  void clear() const
  {
    cuda::fill(stream_, d_face_id, NO_FACE_ID, width * height);
    cuda::fill(stream_, d_ao, NO_AMBIENT_OCCLUSION, width * height);
    CUDA_CHECK_STREAM(stream_);
  }

  void readback(image32i& out_face_ids, image32f& out_ao) const
  {
    out_face_ids = image32i(width, height, 1);
    out_ao = image32f(width, height, 1);
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_face_ids.ptr(), d_face_id, width * height * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_ao.ptr(), d_ao, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_STREAM(stream_);
  }

  ~FramebuffersDevice()
  {
    cudaFreeAsync(d_face_id, stream_);
    cudaFreeAsync(d_ao, stream_);
    cudaStreamSynchronize(stream_);
  }
};
