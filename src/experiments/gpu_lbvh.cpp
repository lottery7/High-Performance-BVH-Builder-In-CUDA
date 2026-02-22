#include "gpu_lbvh.h"

#include <libbase/stats.h>
#include <libbase/timer.h>

#include <filesystem>

#include "../kernels/defines.h"
#include "../utils/cuda_utils.h"
#include "../utils/gpu_wrappers.h"
#include "../utils/utils.h"

// ---------------------------------------------
// Experiment: GPU Ray Tracing with GPU LBVH
// ---------------------------------------------
void runGPULBVH(
    cudaStream_t stream,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters,
    std::vector<double>& out_perf_mrays,
    std::vector<double>& out_build_mtris,
    double& out_build_time_sum,
    double& out_rt_time_sum,
    const image32f& bf_ao,
    const image32i& bf_face_ids,
    bool has_brute_force)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;

  std::vector<double> build_times;
  for (int iter = 0; iter < niters; ++iter) {
    timer t;
    // TODO: построить LBVH на GPU
    build_times.push_back(t.elapsed());
  }

  out_build_time_sum = stats::sum(build_times);
  double build_mtris = scene_gpu.nfaces * 1e-6f / stats::median(build_times);
  std::cout << "GPU LBVH build times (in seconds) - " << stats::valuesStatsLine(build_times) << std::endl;
  std::cout << "GPU LBVH build performance: " << build_mtris << " MTris/s" << std::endl;
  out_build_mtris.push_back(build_mtris);

  fb.clear(stream);

  std::vector<double> rt_times;
  for (int iter = 0; iter < niters; ++iter) {
    timer t;
    // TODO: трассировка лучей с GPU LBVH
    rt_times.push_back(t.elapsed());
  }

  out_rt_time_sum = stats::sum(rt_times);
  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << "GPU with GPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << "GPU with GPU LBVH ray tracing performance: " << mrays << " MRays/s" << std::endl;
  out_perf_mrays.push_back(mrays);

  image32i face_ids;
  image32f ao;
  fb.readback(face_ids, ao);
  saveFramebuffers(results_dir, "with_gpu_lbvh", face_ids, ao);

  if (has_brute_force) {
    validateAgainstBruteForce(bf_ao, bf_face_ids, ao, face_ids, width, height, 3567856512354123ULL, 3453465346387ULL);
  }
}