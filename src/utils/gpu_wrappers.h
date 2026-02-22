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

  unsigned int nvertices = 0;
  unsigned int nfaces = 0;

  SceneGPU(const SceneGeometry& scene, const CameraViewGPU& cam) : nvertices(scene.vertices.size()), nfaces(scene.faces.size())
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&vertices), 3 * nvertices * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&faces), 3 * nfaces * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&camera), sizeof(CameraViewGPU)));

    CUDA_SAFE_CALL(cudaMemcpy(vertices, scene.vertices.data(), 3 * nvertices * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(faces, scene.faces.data(), 3 * nfaces * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(camera, &cam, sizeof(CameraViewGPU), cudaMemcpyHostToDevice));
  }

  ~SceneGPU()
  {
    cudaFree(vertices);
    cudaFree(faces);
    cudaFree(camera);
  }
};

struct FramebuffersGPU {
  int* face_id = nullptr;
  float* ao = nullptr;

  unsigned int width = 0;
  unsigned int height = 0;

  FramebuffersGPU(unsigned int w, unsigned int h) : width(w), height(h)
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&face_id), w * h * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ao), w * h * sizeof(float)));
  }

  void clear(cudaStream_t stream) const
  {
    cuda::fill(stream, face_id, NO_FACE_ID, width * height);
    cuda::fill(stream, ao, NO_AMBIENT_OCCLUSION, width * height);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  void readback(image32i& out_face_ids, image32f& out_ao) const
  {
    out_face_ids = image32i(width, height, 1);
    out_ao = image32f(width, height, 1);
    CUDA_SAFE_CALL(cudaMemcpy(out_face_ids.ptr(), face_id, width * height * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(out_ao.ptr(), ao, width * height * sizeof(float), cudaMemcpyDeviceToHost));
  }

  ~FramebuffersGPU()
  {
    cudaFree(face_id);
    cudaFree(ao);
  }
};

struct LBVHDataGPU {
  BVHNodeGPU* nodes = nullptr;
  unsigned int* leaf_face_indices = nullptr;

  LBVHDataGPU(const std::vector<BVHNodeGPU>& nodes_cpu, const std::vector<uint32_t>& indices_cpu)
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&nodes), nodes_cpu.size() * sizeof(BVHNodeGPU)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&leaf_face_indices), indices_cpu.size() * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(nodes, nodes_cpu.data(), nodes_cpu.size() * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(leaf_face_indices, indices_cpu.data(), indices_cpu.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  }

  ~LBVHDataGPU()
  {
    cudaFree(nodes);
    cudaFree(leaf_face_indices);
  }
};
