#include "my_gpu_lbvh.h"

#include <libbase/stats.h>
#include <thrust/sort.h>

#include <filesystem>

#include "../kernels/defines.h"
#include "../utils/cuda_utils.h"
#include "../utils/device_wrappers.h"
#include "../utils/utils.h"
#include "libbase/timer.h"

#define EXPERIMENT_NAME "My GPU LBVH"

RayTracingResult run_my_gpu_lbvh(
    cudaStream_t stream,
    const SceneDevice& scene_gpu,
    FramebuffersDevice& fb,
    const std::string& results_dir,
    int n_iters)
{
  std::cout << "Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene_gpu.n_faces;

  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_sorted_indices = nullptr;
  BVHNode* d_lbvh_nodes = nullptr;
  int* d_parent = nullptr;
  unsigned int* d_flags = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sorted_indices, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_lbvh_nodes, sizeof(BVHNode) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parent, sizeof(int) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_flags, sizeof(unsigned int) * (n_faces - 1), stream));
  CUDA_CHECK_STREAM(stream);

  std::vector<double> build_times;
  for (int iter = 0; iter < n_iters + WARMUP_ITERS; ++iter) {
    timer bvh_build_t;

    cuda::fill_indices(stream, d_sorted_indices, n_faces);
    cuda::compute_mortons(stream, scene_gpu.d_faces, scene_gpu.d_vertices, n_faces, scene_gpu.aabb, d_morton_codes);
    cuda::sort_by_key(stream, d_morton_codes, d_sorted_indices, n_faces);
    cuda::build_lbvh(stream, d_morton_codes, n_faces, d_lbvh_nodes);
    cuda::build_aabb_leaves(stream, scene_gpu.d_vertices, scene_gpu.d_faces, d_sorted_indices, n_faces, d_lbvh_nodes);
    cuda::build_aabb(stream, n_faces, d_lbvh_nodes, d_parent, d_flags);
    CUDA_CHECK_STREAM(stream);

    if (iter >= WARMUP_ITERS) {
      build_times.push_back(bvh_build_t.elapsed());
    }
  }

  double build_mtris = n_faces * 1e-6f / stats::median(build_times);
  std::cout << EXPERIMENT_NAME " build times (in seconds) - " << stats::valuesStatsLine(build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;
  report_sah(stream, n_faces, d_lbvh_nodes);

  fb.clear();

  std::vector<double> rt_times;
  for (int iter = 0; iter < n_iters + WARMUP_ITERS; ++iter) {
    timer ray_tracing_t;

    cuda::ray_tracing_render_using_bvh(
        stream,
        width,
        height,
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        d_lbvh_nodes,
        d_sorted_indices,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    CUDA_CHECK_STREAM(stream);

    if (iter >= WARMUP_ITERS) {
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

  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_sorted_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_lbvh_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parent, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_flags, stream));
  CUDA_CHECK_STREAM(stream);

  return res;
}