#include <libbase/stats.h>

#include <cfloat>
#include <cub/device/device_radix_sort.cuh>
#include <vector>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/nexus_bvh/nexus_bvh2.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "kernels/hploc/hploc_bvh2.cuh"
#include "kernels/ray_tracing/rt.cuh"
#include "nexus_bvh2.h"

#define EXPERIMENT_NAME "NexusBVH BVH2"

namespace
{
  class StepEvents
  {
   public:
    StepEvents()
    {
      create(total_start);
      create(total_stop);
      create(scene_bounds_start);
      create(scene_bounds_stop);
      create(morton_start);
      create(morton_stop);
      create(sort_start);
      create(sort_stop);
      create(build_start);
      create(build_stop);
      create(rt_start);
      create(rt_stop);
    }

    ~StepEvents()
    {
      destroy(total_start);
      destroy(total_stop);
      destroy(scene_bounds_start);
      destroy(scene_bounds_stop);
      destroy(morton_start);
      destroy(morton_stop);
      destroy(sort_start);
      destroy(sort_stop);
      destroy(build_start);
      destroy(build_stop);
      destroy(rt_start);
      destroy(rt_stop);
    }

    StepEvents(const StepEvents&) = delete;
    StepEvents& operator=(const StepEvents&) = delete;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t scene_bounds_start = nullptr;
    cudaEvent_t scene_bounds_stop = nullptr;
    cudaEvent_t morton_start = nullptr;
    cudaEvent_t morton_stop = nullptr;
    cudaEvent_t sort_start = nullptr;
    cudaEvent_t sort_stop = nullptr;
    cudaEvent_t build_start = nullptr;
    cudaEvent_t build_stop = nullptr;
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

RayTracingResult run_nexus_bvh2(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_nodes = 2 * n_faces - 1;

  BVH2Node* d_bvh = nullptr;
  cuda::nexus_bvh::Workspace workspace;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVH2Node) * max_nodes, stream));
  cuda::nexus_bvh::allocate_workspace(stream, workspace, n_faces);
  CUDA_SYNC_STREAM(stream);

  cuda::nexus_bvh::BuildState build_state{};
  build_state.scene_bounds = workspace.d_scene_bounds;
  build_state.nodes = d_bvh;
  build_state.cluster_indices = workspace.d_cluster_indices;
  build_state.parent_indices = workspace.d_parent_indices;
  build_state.prim_count = n_faces;
  build_state.cluster_count = workspace.d_cluster_count;

  const AABB empty_scene_bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

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

    build_state.cluster_indices = workspace.d_cluster_indices;
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.scene_bounds, &empty_scene_bounds, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(build_state.parent_indices, 0xff, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.cluster_count, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.scene_bounds_start, stream));
    cuda::nexus_bvh::compute_scene_bounds_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, scene.d_faces, scene.d_vertices);
    CUDA_SAFE_CALL(cudaEventRecord(events.scene_bounds_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.morton_start, stream));
    cuda::nexus_bvh::compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, workspace.d_morton_codes);
    CUDA_SAFE_CALL(cudaEventRecord(events.morton_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.sort_start, stream));
    cub::DeviceRadixSort::SortPairs(
        workspace.d_sort_temp_storage,
        workspace.sort_temp_storage_bytes,
        workspace.d_morton_codes,
        workspace.d_morton_codes_sorted,
        workspace.d_cluster_indices,
        workspace.d_cluster_indices_sorted,
        n_faces,
        0,
        32,
        stream);
    CUDA_SAFE_CALL(cudaEventRecord(events.sort_stop, stream));

    build_state.cluster_indices = workspace.d_cluster_indices_sorted;
    constexpr unsigned int block_size = 64;
    CUDA_SAFE_CALL(cudaEventRecord(events.build_start, stream));
    cuda::nexus_bvh::build_bvh2_kernel<<<div_ceil(static_cast<int>(n_faces), static_cast<int>(block_size)), block_size, 0, stream>>>(
        build_state,
        workspace.d_morton_codes_sorted);
    CUDA_SAFE_CALL(cudaEventRecord(events.build_stop, stream));

    fb.clear();

    CUDA_SAFE_CALL(cudaEventRecord(events.rt_start, stream));
    cuda::hploc::rt_hploc_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        d_bvh,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.rt_stop, stream));
    CUDA_SAFE_CALL(cudaEventRecord(events.total_stop, stream));

    CUDA_SAFE_CALL(cudaEventSynchronize(events.total_stop));

    const double scene_bounds_ms = elapsed_ms(events.scene_bounds_start, events.scene_bounds_stop);
    const double morton_ms = elapsed_ms(events.morton_start, events.morton_stop);
    const double sort_ms = elapsed_ms(events.sort_start, events.sort_stop);
    const double build_ms = elapsed_ms(events.build_start, events.build_stop);
    const double build_pipeline_ms = scene_bounds_ms + morton_ms + sort_ms + build_ms;
    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      scene_bounds_times.push_back(scene_bounds_ms);
      morton_times.push_back(morton_ms);
      sort_times.push_back(sort_ms);
      build_kernel_times.push_back(build_ms);
      build_times.push_back(build_pipeline_ms);
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
  save_framebuffers(results_dir, "with_nexus_bvh", result.face_ids, result.ao);

  cuda::nexus_bvh::free_workspace(stream, workspace);
  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
