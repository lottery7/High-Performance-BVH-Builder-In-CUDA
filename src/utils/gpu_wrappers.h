#pragma once

#include <cuda_runtime_api.h>
#include <libimages/debug_io.h>

#include "../io/scene_reader.h"
#include "../kernels/defines.h"
#include "../kernels/kernels.h"
#include "../utils/cuda_utils.h"

struct SceneGPU {
  float* vertices = nullptr;
  unsigned int* faces = nullptr;
  CameraViewGPU* camera = nullptr;

  cudaStream_t stream_;
  unsigned int nvertices = 0;
  unsigned int nfaces = 0;
  point3f cMin;
  point3f cMax;

  SceneGPU(cudaStream_t stream, const SceneGeometry& scene, const CameraViewGPU& cam)
      : stream_(stream), nvertices(scene.vertices.size()), nfaces(scene.faces.size()), cMin(scene.cMin), cMax(scene.cMax)
  {
    static_assert(sizeof(point3f) == 12);
    static_assert(sizeof(point3u) == 12);

    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&vertices), 3 * nvertices * sizeof(float), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&faces), 3 * nfaces * sizeof(unsigned int), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&camera), sizeof(CameraViewGPU), stream_));
    CUDA_CHECK_STREAM(stream_);

    CUDA_SAFE_CALL(cudaMemcpyAsync(vertices, scene.vertices.data(), 3 * nvertices * sizeof(float), cudaMemcpyHostToDevice, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(faces, scene.faces.data(), 3 * nfaces * sizeof(unsigned int), cudaMemcpyHostToDevice, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(camera, &cam, sizeof(CameraViewGPU), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK_STREAM(stream_);
  }

  ~SceneGPU()
  {
    cudaFreeAsync(vertices, stream_);
    cudaFreeAsync(faces, stream_);
    cudaFreeAsync(camera, stream_);
    cudaStreamSynchronize(stream_);
  }
};

struct FramebuffersGPU {
  int* face_id = nullptr;
  float* ao = nullptr;

  cudaStream_t stream_;
  unsigned int width = 0;
  unsigned int height = 0;

  FramebuffersGPU(cudaStream_t stream, unsigned int w, unsigned int h) : stream_(stream), width(w), height(h)
  {
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&face_id), w * h * sizeof(int), stream_));
    CUDA_SAFE_CALL(cudaMallocAsync(reinterpret_cast<void**>(&ao), w * h * sizeof(float), stream_));
    CUDA_CHECK_STREAM(stream_);
  }

  void clear() const
  {
    cuda::fill(stream_, face_id, NO_FACE_ID, width * height);
    cuda::fill(stream_, ao, NO_AMBIENT_OCCLUSION, width * height);
    CUDA_CHECK_STREAM(stream_);
  }

  void readback(image32i& out_face_ids, image32f& out_ao) const
  {
    out_face_ids = image32i(width, height, 1);
    out_ao = image32f(width, height, 1);
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_face_ids.ptr(), face_id, width * height * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_SAFE_CALL(cudaMemcpyAsync(out_ao.ptr(), ao, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK_STREAM(stream_);
  }

  ~FramebuffersGPU()
  {
    cudaFreeAsync(face_id, stream_);
    cudaFreeAsync(ao, stream_);
    cudaStreamSynchronize(stream_);
  }
};
