#include <libbase/stats.h>
#include <thrust/device_ptr.h>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/detail/sequence.inl>
#include <vector>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/lbvh/lbvh.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "kernels/ray_tracing/rt.cuh"
#include "lbvh.h"

#define EXPERIMENT_NAME "LBVH"

namespace
{
  class StepEvents
  {
   public:
    StepEvents()
    {
      create(total_start);
      create(total_stop);
      create(fill_indices_start);
      create(fill_indices_stop);
      create(morton_start);
      create(morton_stop);
      create(sort_start);
      create(sort_stop);
      create(hierarchy_start);
      create(hierarchy_stop);
      create(leaf_aabb_start);
      create(leaf_aabb_stop);
      create(parent_start);
      create(parent_stop);
      create(internal_aabb_start);
      create(internal_aabb_stop);
      create(rt_start);
      create(rt_stop);
    }

    ~StepEvents()
    {
      destroy(total_start);
      destroy(total_stop);
      destroy(fill_indices_start);
      destroy(fill_indices_stop);
      destroy(morton_start);
      destroy(morton_stop);
      destroy(sort_start);
      destroy(sort_stop);
      destroy(hierarchy_start);
      destroy(hierarchy_stop);
      destroy(leaf_aabb_start);
      destroy(leaf_aabb_stop);
      destroy(parent_start);
      destroy(parent_stop);
      destroy(internal_aabb_start);
      destroy(internal_aabb_stop);
      destroy(rt_start);
      destroy(rt_stop);
    }

    StepEvents(const StepEvents&) = delete;
    StepEvents& operator=(const StepEvents&) = delete;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t fill_indices_start = nullptr;
    cudaEvent_t fill_indices_stop = nullptr;
    cudaEvent_t morton_start = nullptr;
    cudaEvent_t morton_stop = nullptr;
    cudaEvent_t sort_start = nullptr;
    cudaEvent_t sort_stop = nullptr;
    cudaEvent_t hierarchy_start = nullptr;
    cudaEvent_t hierarchy_stop = nullptr;
    cudaEvent_t leaf_aabb_start = nullptr;
    cudaEvent_t leaf_aabb_stop = nullptr;
    cudaEvent_t parent_start = nullptr;
    cudaEvent_t parent_stop = nullptr;
    cudaEvent_t internal_aabb_start = nullptr;
    cudaEvent_t internal_aabb_stop = nullptr;
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

RayTracingResult run_lbvh(cudaStream_t stream, const cuda::Scene& scene_gpu, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene_gpu.n_faces;
  const unsigned int n_nodes = 2 * n_faces - 1;

  BVHNode* d_bvh = nullptr;
  MortonCode* d_morton_codes = nullptr;
  MortonCode* d_morton_codes_sorted = nullptr;
  unsigned int* d_indices = nullptr;
  unsigned int* d_indices_sorted = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_flags = nullptr;
  void* d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVHNode) * n_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes_sorted, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_indices, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_indices_sorted, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * n_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_flags, sizeof(unsigned int) * (n_faces - 1), stream));
  cub::DoubleBuffer<MortonCode> morton_codes_buffer(d_morton_codes, d_morton_codes_sorted);
  cub::DoubleBuffer<unsigned int> indices_buffer(d_indices, d_indices_sorted);
  cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_storage_bytes, morton_codes_buffer, indices_buffer, static_cast<int>(n_faces), 0, 32, stream);
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sort_temp_storage, sort_temp_storage_bytes, stream));
  CUDA_SYNC_STREAM(stream);

  std::vector<double> fill_indices_times;
  std::vector<double> morton_times;
  std::vector<double> sort_times;
  std::vector<double> hierarchy_times;
  std::vector<double> leaf_aabb_times;
  std::vector<double> parent_times;
  std::vector<double> internal_aabb_times;
  std::vector<double> build_pipeline_times;
  std::vector<double> rt_times;
  std::vector<double> total_times;

  StepEvents events;

  const AdaptiveWarmupResult pipeline_warmup = benchmark::run_adaptive([&](bool collect) {
    CUDA_SAFE_CALL(cudaEventRecord(events.total_start, stream));

    auto d_ptr_indices = thrust::device_pointer_cast(d_indices);
    CUDA_SAFE_CALL(cudaEventRecord(events.fill_indices_start, stream));
    thrust::sequence(thrust::cuda::par_nosync.on(stream), d_ptr_indices, d_ptr_indices + n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.fill_indices_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.morton_start, stream));
    cuda::compute_mortons_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene_gpu.aabb,
        scene_gpu.d_faces,
        scene_gpu.d_vertices,
        d_morton_codes,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.morton_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.sort_start, stream));
    cub::DoubleBuffer<MortonCode> morton_codes_buffer(d_morton_codes, d_morton_codes_sorted);
    cub::DoubleBuffer<unsigned int> indices_buffer(d_indices, d_indices_sorted);
    cub::DeviceRadixSort::SortPairs(
        d_sort_temp_storage,
        sort_temp_storage_bytes,
        morton_codes_buffer,
        indices_buffer,
        static_cast<int>(n_faces),
        0,
        32,
        stream);
    if (morton_codes_buffer.Current() != d_morton_codes) {
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_morton_codes, morton_codes_buffer.Current(), sizeof(MortonCode) * n_faces, cudaMemcpyDeviceToDevice, stream));
    }
    if (indices_buffer.Current() != d_indices) {
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_indices, indices_buffer.Current(), sizeof(unsigned int) * n_faces, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_SAFE_CALL(cudaEventRecord(events.sort_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.hierarchy_start, stream));
    cuda::lbvh::build_bvh_kernel<<<compute_grid(n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(d_bvh, d_morton_codes, n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.hierarchy_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.leaf_aabb_start, stream));
    cuda::lbvh::build_aabb_leaves_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        d_bvh,
        scene_gpu.d_faces,
        scene_gpu.d_vertices,
        d_indices,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.leaf_aabb_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.parent_start, stream));
    cuda::lbvh::compute_parents_kernel<<<compute_grid(n_faces - 1), DEFAULT_GROUP_SIZE, 0, stream>>>(d_bvh, d_parents, n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.parent_stop, stream));

    CUDA_SAFE_CALL(cudaMemsetAsync(d_flags, 0, sizeof(unsigned int) * (n_faces - 1), stream));
    CUDA_SAFE_CALL(cudaEventRecord(events.internal_aabb_start, stream));
    cuda::lbvh::build_aabb_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(d_bvh, d_parents, d_flags, n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.internal_aabb_stop, stream));

    fb.clear();

    CUDA_SAFE_CALL(cudaEventRecord(events.rt_start, stream));
    cuda::lbvh::rt_lbvh_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene_gpu.d_vertices,
        scene_gpu.d_faces,
        d_bvh,
        d_indices,
        fb.d_face_id,
        fb.d_ao,
        scene_gpu.d_camera,
        scene_gpu.n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.rt_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.total_stop, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(events.total_stop));

    const double fill_indices_ms = elapsed_ms(events.fill_indices_start, events.fill_indices_stop);
    const double morton_ms = elapsed_ms(events.morton_start, events.morton_stop);
    const double sort_ms = elapsed_ms(events.sort_start, events.sort_stop);
    const double hierarchy_ms = elapsed_ms(events.hierarchy_start, events.hierarchy_stop);
    const double leaf_aabb_ms = elapsed_ms(events.leaf_aabb_start, events.leaf_aabb_stop);
    const double parent_ms = elapsed_ms(events.parent_start, events.parent_stop);
    const double internal_aabb_ms = elapsed_ms(events.internal_aabb_start, events.internal_aabb_stop);
    const double build_pipeline_ms = fill_indices_ms + morton_ms + sort_ms + hierarchy_ms + leaf_aabb_ms + parent_ms + internal_aabb_ms;
    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      fill_indices_times.push_back(fill_indices_ms);
      morton_times.push_back(morton_ms);
      sort_times.push_back(sort_ms);
      hierarchy_times.push_back(hierarchy_ms);
      leaf_aabb_times.push_back(leaf_aabb_ms);
      parent_times.push_back(parent_ms);
      internal_aabb_times.push_back(internal_aabb_ms);
      build_pipeline_times.push_back(build_pipeline_ms);
      rt_times.push_back(rt_ms);
      total_times.push_back(total_ms);
    }

    return total_ms;
  });
  print_warmup_report(EXPERIMENT_NAME, pipeline_warmup);

  const double build_mtris = n_faces * 1e-3 / stats::median(build_pipeline_times);
  std::cout << EXPERIMENT_NAME " fill indices times (in ms) - " << stats::median(fill_indices_times) << std::endl;
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << stats::median(sort_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build hierarchy kernel times (in ms) - " << stats::median(hierarchy_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build leaf aabb times (in ms) - " << stats::median(leaf_aabb_times) << std::endl;
  std::cout << EXPERIMENT_NAME " compute parents times (in ms) - " << stats::median(parent_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build internal aabb times (in ms) - " << stats::median(internal_aabb_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << stats::median(build_pipeline_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;
  report_sah(stream, d_bvh, n_nodes);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << stats::median(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << stats::median(total_times) << std::endl;

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_lbvh", res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_indices, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_indices_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_flags, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_sort_temp_storage, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}
