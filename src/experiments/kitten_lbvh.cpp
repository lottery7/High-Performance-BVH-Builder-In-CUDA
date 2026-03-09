#include "kitten_lbvh.h"

#include <filesystem>

#include "../utils/defines.h"
#include "../utils/utils.h"
#include "KittenEngine/includes/modules/Bound.h"
#include "KittenGpuLBVH/lbvh.cuh"
#include "libbase/stats.h"
#include "libbase/timer.h"

// https://github.com/jerry060599/KittenGpuLBVH
#define EXPERIMENT_NAME "Kitten LBVH"

RayTracingResult run_kitten_lbvh(const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene_gpu.n_faces;

  Kitten::Bound<3, float>* d_face_bounds;
  d_face_bounds = nullptr;
  BVHNode* d_bvh = nullptr;
  unsigned int* d_sorted_indices = nullptr;

  CUDA_SAFE_CALL(cudaMalloc(&d_face_bounds, n_faces * sizeof(Kitten::Bound<3, float>)));
  CUDA_SAFE_CALL(cudaMalloc(&d_bvh, (2 * n_faces - 1) * sizeof(BVHNode)));
  CUDA_SAFE_CALL(cudaMalloc(&d_sorted_indices, sizeof(unsigned int) * n_faces));

  Kitten::LBVH lbvh;
  std::vector<double> build_times;
  for (int iter = 0; iter < BENCHMARK_ITERS + WARMUP_ITERS; ++iter) {
    timer build_timer;  // use host-size timer because Kitten LBVH works with default device stream

    cuda::compute_faces_aabb(0, scene_gpu.d_vertices, scene_gpu.d_faces, d_face_bounds, n_faces);
    lbvh.compute(d_face_bounds, n_faces);
    cuda::convert_to_bvh_nodes(0, lbvh.internalNodes(), lbvh.objects(), lbvh.objectIds(), d_bvh, d_sorted_indices, n_faces);
    CUDA_SYNC_STREAM(0);

    if (iter >= WARMUP_ITERS) {
      build_times.push_back(build_timer.elapsed());
    }
  }
  lbvh.bvhSelfCheck();

  double build_mtris = n_faces * 1e-6f / stats::median(build_times);
  std::cout << EXPERIMENT_NAME " build times (in seconds) - " << stats::valuesStatsLine(build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;
  report_sah(0, d_bvh, n_faces);

  fb.clear();

  std::vector<double> rt_times;
  for (int iter = 0; iter < BENCHMARK_ITERS + WARMUP_ITERS; ++iter) {
    timer rt_timer;
    cuda::ray_tracing_render_using_bvh(
        0,
        width,
        height,
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        d_bvh,
        d_sorted_indices,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    CUDA_SYNC_STREAM(0);
    if (iter >= WARMUP_ITERS) {
      rt_times.push_back(rt_timer.elapsed());
    }
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total frame time: " << stats::median(build_times) + stats::median(rt_times) << " seconds" << std::endl;

  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_kitten_lbvh", res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFree(d_face_bounds));
  CUDA_SAFE_CALL(cudaFree(d_bvh));

  return res;
}