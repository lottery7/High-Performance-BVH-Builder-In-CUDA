#include <libbase/stats.h>
#include <thrust/device_ptr.h>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/detail/sequence.inl>
#include <vector>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc.cuh"
#include "../kernels/ray_tracing/rt.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "hploc.h"

#define EXPERIMENT_NAME "H-PLOC"

namespace
{
  class StepEvents
  {
   public:
    StepEvents()
    {
      create(total_start);
      create(total_stop);
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
    float elapsed_ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
  }
}  // namespace

RayTracingResult run_hploc(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int n_nodes_capacity = 2 * n_faces - 1;

  BVHNode* d_bvh = nullptr;
  MortonCode* d_morton_codes = nullptr;
  MortonCode* d_morton_codes_sorted = nullptr;
  unsigned int* d_cluster_ids = nullptr;
  unsigned int* d_cluster_ids_sorted = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_n_clusters = nullptr;
  void* d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVHNode) * n_nodes_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes_sorted, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids, sizeof(unsigned int) * n_nodes_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids_sorted, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * n_nodes_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_clusters, sizeof(unsigned int), stream));
  cub::DeviceRadixSort::SortPairs(
      d_sort_temp_storage,
      sort_temp_storage_bytes,
      d_morton_codes,
      d_morton_codes_sorted,
      d_cluster_ids,
      d_cluster_ids_sorted,
      static_cast<int>(n_faces),
      2,
      32,
      stream);
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sort_temp_storage, sort_temp_storage_bytes, stream));
  CUDA_SYNC_STREAM(stream);

  std::vector<double> morton_times;
  std::vector<double> sort_times;
  std::vector<double> build_kernel_times;
  std::vector<double> build_pipeline_times;
  std::vector<double> rt_times;
  std::vector<double> total_times;

  morton_times.reserve(benchmark_iters());
  sort_times.reserve(benchmark_iters());
  build_kernel_times.reserve(benchmark_iters());
  build_pipeline_times.reserve(benchmark_iters());
  rt_times.reserve(benchmark_iters());
  total_times.reserve(benchmark_iters());

  StepEvents events;

  const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
    CUDA_SAFE_CALL(cudaEventRecord(events.total_start, stream));

    CUDA_SAFE_CALL(cudaMemsetAsync(d_parents, INVALID_INDEX, sizeof(unsigned int) * n_nodes_capacity, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_n_clusters, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));

    cuda::hploc::build_leaves_nodes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(scene.d_faces, n_faces, scene.d_vertices, d_bvh);
    cuda::fill_indices(stream, d_cluster_ids, n_faces);

    CUDA_SAFE_CALL(cudaEventRecord(events.morton_start, stream));
    cuda::compute_mortons_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene.aabb,
        scene.d_faces,
        scene.d_vertices,
        d_morton_codes,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.morton_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.sort_start, stream));
    cub::DeviceRadixSort::SortPairs(
        d_sort_temp_storage,
        sort_temp_storage_bytes,
        d_morton_codes,
        d_morton_codes_sorted,
        d_cluster_ids,
        d_cluster_ids_sorted,
        static_cast<int>(n_faces),
        2,
        32,
        stream);
    CUDA_SAFE_CALL(cudaEventRecord(events.sort_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.build_start, stream));
    constexpr size_t block_size = 128;
    cuda::hploc::build_kernel<<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(
        d_parents,
        d_morton_codes_sorted,
        d_bvh,
        d_cluster_ids_sorted,
        d_n_clusters,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.build_stop, stream));

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

    const double morton_ms = elapsed_ms(events.morton_start, events.morton_stop);
    const double sort_ms = elapsed_ms(events.sort_start, events.sort_stop);
    const double build_ms = elapsed_ms(events.build_start, events.build_stop);
    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      morton_times.push_back(morton_ms);
      sort_times.push_back(sort_ms);
      build_kernel_times.push_back(build_ms);
      build_pipeline_times.push_back(morton_ms + sort_ms + build_ms);
      rt_times.push_back(rt_ms);
      total_times.push_back(total_ms);
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / stats::median(build_pipeline_times);
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << stats::median(build_kernel_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << stats::median(build_pipeline_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_nodes;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_nodes, d_n_clusters, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_nodes && n_nodes < n_nodes_capacity + 1, 541056);
  report_sah_hploc(stream, d_bvh, n_nodes, n_faces);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << stats::median(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << stats::median(total_times) << std::endl;

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_clusters, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_sort_temp_storage, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}
