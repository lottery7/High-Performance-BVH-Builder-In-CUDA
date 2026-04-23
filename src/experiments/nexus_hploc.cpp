#include "nexus_hploc.h"

#include <libbase/stats.h>

#include <vector>

#include "../kernels/kernels.h"
#include "../kernels/nexus_hploc/nexus_hploc.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"

#define EXPERIMENT_NAME "NexusBVH H-PLOC"

namespace
{
  class StepEvents
  {
   public:
    StepEvents()
    {
      create(total_start);
      create(total_stop);
      create(rt_start);
      create(rt_stop);
    }

    ~StepEvents()
    {
      destroy(total_start);
      destroy(total_stop);
      destroy(rt_start);
      destroy(rt_stop);
    }

    StepEvents(const StepEvents&) = delete;
    StepEvents& operator=(const StepEvents&) = delete;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t rt_start = nullptr;
    cudaEvent_t rt_stop = nullptr;

   private:
    static void create(cudaEvent_t& event) { CUDA_SAFE_CALL(cudaEventCreate(&event)); }

    static void destroy(cudaEvent_t event)
    {
      if (event != nullptr) cudaEventDestroy(event);
    }
  };

  float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
  {
    float elapsed = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
  }
}  // namespace

RayTracingResult run_nexus_hploc(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_nodes = 2 * n_faces - 1;

  BVHNode* d_bvh = nullptr;
  cuda::nexus_hploc::Workspace workspace;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVHNode) * max_nodes, stream));
  cuda::nexus_hploc::allocate_workspace(stream, workspace, n_faces);
  CUDA_SYNC_STREAM(stream);

  std::vector<double> scene_bounds_times;
  std::vector<double> morton_times;
  std::vector<double> sort_times;
  std::vector<double> build_kernel_times;
  std::vector<double> build_times;
  std::vector<double> rt_times;
  std::vector<double> total_times;
  StepEvents events;

  const AdaptiveWarmupResult build_warmup = benchmark::run_adaptive([&](bool collect) {
    CUDA_SAFE_CALL(cudaEventRecord(events.total_start, stream));
    cuda::nexus_hploc::BuildTimings build_timings;
    cuda::nexus_hploc::build(stream, scene.d_faces, scene.d_vertices, d_bvh, workspace, n_faces, build_timings);

    CUDA_SAFE_CALL(cudaEventRecord(events.rt_start, stream));
    cuda::rt_hploc(stream, width, height, scene.d_vertices, scene.d_faces, d_bvh, fb.d_face_id, fb.d_ao, scene.d_camera, n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.rt_stop, stream));
    CUDA_SAFE_CALL(cudaEventRecord(events.total_stop, stream));

    CUDA_SAFE_CALL(cudaEventSynchronize(events.total_stop));

    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      scene_bounds_times.push_back(build_timings.scene_bounds_ms);
      morton_times.push_back(build_timings.morton_ms);
      sort_times.push_back(build_timings.sort_ms);
      build_kernel_times.push_back(build_timings.build_ms);
      build_times.push_back(build_timings.build_pipeline_ms);
      rt_times.push_back(rt_ms);
      total_times.push_back(total_ms);
    }
    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, build_warmup);

  const double build_mtris = n_faces * 1e-3 / stats::median(build_times);
  std::cout << EXPERIMENT_NAME " compute scene bounds times (in ms) - " << stats::median(scene_bounds_times) << std::endl;
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << stats::median(sort_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << stats::median(build_kernel_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << stats::median(build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_nodes, workspace.d_cluster_count, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(n_nodes == max_nodes, 226786314);
  report_sah_hploc(stream, d_bvh, max_nodes, n_faces);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << stats::median(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << stats::median(total_times) << std::endl;

  RayTracingResult result;
  fb.readback(result.face_ids, result.ao);
  save_framebuffers(results_dir, "with_nexus_hploc", result.face_ids, result.ao);

  cuda::nexus_hploc::free_workspace(stream, workspace);
  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
