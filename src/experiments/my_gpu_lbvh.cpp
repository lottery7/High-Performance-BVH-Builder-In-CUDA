#include "my_gpu_lbvh.h"

#include <libbase/stats.h>

#include <filesystem>

#include "../kernels/my_lbvh/my_lbvh.h"
#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "libbase/timer.h"

#define EXPERIMENT_NAME "My GPU LBVH"

RayTracingResult run_my_gpu_lbvh(cudaStream_t stream, const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene_gpu.n_faces;
  const unsigned int n_nodes = 2 * n_faces - 1;

  BVHNode* d_bvh = nullptr;
  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_indices = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_flags = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVHNode) * n_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_indices, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(int) * n_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_flags, sizeof(unsigned int) * (n_faces - 1), stream));
  CUDA_SYNC_STREAM(stream);

  const int warmup = warmup_iters();
  const int benchmark = benchmark_iters();

  std::vector<double> build_times;
  for (int iter = 0; iter < benchmark + warmup; ++iter) {
    timer bvh_build_t;

    cuda::my_lbvh::build(
        stream,
        scene_gpu.aabb,
        scene_gpu.d_faces,
        scene_gpu.d_vertices,
        d_bvh,
        d_morton_codes,
        d_indices,
        d_parents,
        d_flags,
        n_faces);
    CUDA_SYNC_STREAM(stream);

    if (iter >= warmup) {
      build_times.push_back(bvh_build_t.elapsed());
    }
  }

  double build_mtris = n_faces * 1e-6f / stats::median(build_times);
  std::cout << EXPERIMENT_NAME " build times (in seconds) - " << stats::valuesStatsLine(build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;
  report_sah(stream, d_bvh, n_nodes);

  fb.clear();

  std::vector<double> rt_times;
  for (int iter = 0; iter < benchmark + warmup; ++iter) {
    timer ray_tracing_t;

    cuda::rt_lbvh(
        stream,
        width,
        height,
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        d_bvh,
        d_indices,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    CUDA_SYNC_STREAM(stream);

    if (iter >= warmup) {
      rt_times.push_back(ray_tracing_t.elapsed());
    }
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total frame time: " << stats::median(build_times) + stats::median(rt_times) << " seconds" << std::endl;

  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_gpu_lbvh", res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_flags, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}
