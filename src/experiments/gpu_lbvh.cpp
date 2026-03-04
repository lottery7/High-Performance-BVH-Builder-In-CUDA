#include "gpu_lbvh.h"

#include <libbase/stats.h>
#include <libbase/timer.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <filesystem>

#include "../kernels/defines.h"
#include "../utils/cuda_utils.h"
#include "../utils/gpu_wrappers.h"
#include "../utils/utils.h"

// ---------------------------------------------
// Experiment: GPU Ray Tracing with GPU LBVH
// ---------------------------------------------
RayTracingResult runGPULBVH(cudaStream_t stream, const SceneGPU& scene_gpu, FramebuffersGPU& fb, const std::string& results_dir, int niters)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int nfaces = scene_gpu.nfaces;

  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_sorted_indices = nullptr;
  BVHNodeGPU* d_lbvh_nodes = nullptr;
  int* d_parent = nullptr;
  unsigned int* d_flags = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * nfaces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sorted_indices, sizeof(unsigned int) * nfaces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_lbvh_nodes, sizeof(BVHNodeGPU) * (2 * nfaces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parent, sizeof(int) * (2 * nfaces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_flags, sizeof(unsigned int) * (nfaces - 1), stream));

  CUDA_CHECK_STREAM(stream);

  float cMinX = scene_gpu.cMin.x;
  float cMinY = scene_gpu.cMin.y;
  float cMinZ = scene_gpu.cMin.z;

  float cMaxX = scene_gpu.cMax.x;
  float cMaxY = scene_gpu.cMax.y;
  float cMaxZ = scene_gpu.cMax.z;

  std::vector<double> build_times;
  for (int iter = 0; iter < niters; ++iter) {
    cuda::CudaTimer cuda_timer(stream);

    cuda::fill_indices(stream, d_sorted_indices, nfaces);
    cuda::compute_mortons(stream, scene_gpu.faces, scene_gpu.vertices, nfaces, cMinX, cMinY, cMinZ, cMaxX, cMaxY, cMaxZ, d_morton_codes);
    cuda::sort_by_key(stream, d_morton_codes, d_sorted_indices, nfaces);
    cuda::build_lbvh(stream, d_morton_codes, nfaces, d_lbvh_nodes);
    cuda::build_aabb_leaves(stream, scene_gpu.vertices, scene_gpu.faces, d_sorted_indices, nfaces, d_lbvh_nodes);
    cuda::build_aabb(stream, nfaces, d_lbvh_nodes, d_parent, d_flags);
    CUDA_CHECK_STREAM(stream);

    build_times.push_back(cuda_timer.elapsed());
  }

  double build_mtris = nfaces * 1e-6f / stats::median(build_times);
  std::cout << "GPU LBVH build times (in seconds) - " << stats::valuesStatsLine(build_times) << std::endl;
  std::cout << "GPU LBVH build performance: " << build_mtris << " MTris/s" << std::endl;

  // ---------- Ray tracing with GPU LBVH (timed) ----------
  fb.clear(stream);

  std::vector<double> rt_times;
  for (int iter = 0; iter < niters; ++iter) {
    cuda::CudaTimer cuda_timer(stream);
    cuda::ray_tracing_render_using_bvh(
        stream,
        dim3(divCeil(width, 16), divCeil(height, 16)),
        dim3(16, 16),
        scene_gpu.vertices,
        scene_gpu.faces,
        d_lbvh_nodes,
        d_sorted_indices,
        fb.face_id,
        fb.ao,
        scene_gpu.camera,
        scene_gpu.nfaces);
    CUDA_CHECK_STREAM(stream);
    rt_times.push_back(cuda_timer.elapsed());
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << "GPU with GPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << "GPU with GPU LBVH ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << "GPU with GPU LBVH total frame time: " << stats::median(build_times) + stats::median(rt_times) << " seconds" << std::endl;

  // ---------- Readback ----------
  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  saveFramebuffers(results_dir, "with_gpu_lbvh", res.face_ids, res.ao);

  // ---------- Free ----------
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_sorted_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_lbvh_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parent, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_flags, stream));
  CUDA_SAFE_CALL(cudaStreamSynchronize(stream));

  return res;
}