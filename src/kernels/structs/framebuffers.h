#pragma once

#include <cuda_runtime_api.h>
#include <libimages/debug_io.h>

#include "../../io/scene_reader.h"
#include "../../kernels/defines.h"
#include "../../kernels/kernels.h"
#include "../../utils/cuda_utils.h"

namespace cuda
{
  class Framebuffers
  {
    cudaStream_t stream_;

   public:
    int* d_face_id = nullptr;
    float* d_ao = nullptr;

    unsigned int width = 0;
    unsigned int height = 0;

    Framebuffers(cudaStream_t stream, unsigned int w, unsigned int h) : stream_(stream), width(w), height(h)
    {
      CUDA_SAFE_CALL(cudaMallocAsync(&d_face_id, w * h * sizeof(int), stream_));
      CUDA_SAFE_CALL(cudaMallocAsync(&d_ao, w * h * sizeof(float), stream_));
    }

    void clear() const
    {
      cuda::fill(stream_, d_face_id, NO_FACE_ID, width * height);
      cuda::fill(stream_, d_ao, NO_AMBIENT_OCCLUSION, width * height);
    }

    void readback(image32i& out_face_ids, image32f& out_ao) const
    {
      out_face_ids = image32i(width, height, 1);
      out_ao = image32f(width, height, 1);
      CUDA_SAFE_CALL(cudaMemcpyAsync(out_face_ids.ptr(), d_face_id, width * height * sizeof(int), cudaMemcpyDeviceToHost, stream_));
      CUDA_SAFE_CALL(cudaMemcpyAsync(out_ao.ptr(), d_ao, width * height * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }

    ~Framebuffers()
    {
      cudaFreeAsync(d_face_id, stream_);
      cudaFreeAsync(d_ao, stream_);
    }
  };
}  // namespace cuda
