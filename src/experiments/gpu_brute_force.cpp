#include "gpu_brute_force.h"

#include <iostream>

#include "../kernels/kernels.h"
#include "../utils/gpu_wrappers.h"
#include "../utils/utils.h"
#include "libbase/stats.h"
#include "libbase/timer.h"

// ---------------------------------------------
// Experiment: GPU Ray Tracing without BVH (Brute Force)
// ---------------------------------------------
BruteForceResult runBruteForce(cudaStream_t stream, const SceneGPU& scene_gpu, const FramebuffersGPU& fb, const std::string& results_dir, int niters)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;

  std::vector<double> times;
  for (int iter = 0; iter < niters; ++iter) {
    timer t;
    cuda::ray_tracing_render_brute_force(
        stream,
        dim3(divCeil(width, 16), divCeil(height, 16)),
        dim3(16, 16),
        scene_gpu.vertices,
        scene_gpu.faces,
        fb.face_id,
        fb.ao,
        scene_gpu.camera,
        scene_gpu.nfaces);
    times.push_back(t.elapsed());
  }

  std::cout << "GPU brute force ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

  BruteForceResult result;
  result.total_time = stats::sum(times);
  fb.readback(result.face_ids, result.ao);

  rassert(countNonEmpty(result.face_ids, NO_FACE_ID) > width * height / 10, 2345123412);
  rassert(countNonEmpty(result.ao, NO_AMBIENT_OCCLUSION) > width * height / 10, 3423413421);

  saveFramebuffers(results_dir, "brute_force", result.face_ids, result.ao);
  return result;
}
