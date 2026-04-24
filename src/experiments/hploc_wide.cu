#include <libbase/stats.h>
#include <thrust/device_ptr.h>

#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <string>
#include <thrust/detail/sequence.inl>
#include <vector>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/hploc/hploc.cuh"
#include "../kernels/hploc/hploc_wide.cuh"
#include "../kernels/structs/wide_bvh_node.h"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "benchmark.h"
#include "hploc_wide.h"
#include "kernels/ray_tracing/rt.cuh"

namespace
{
  class StepEvents
  {
   public:
    StepEvents()
    {
      create(total_start);
      create(total_stop);
      create(build_leaves_start);
      create(build_leaves_stop);
      create(fill_indices_start);
      create(fill_indices_stop);
      create(morton_start);
      create(morton_stop);
      create(sort_start);
      create(sort_stop);
      create(binary_build_start);
      create(binary_build_stop);
      create(conversion_start);
      create(conversion_stop);
      create(rt_start);
      create(rt_stop);
    }

    ~StepEvents()
    {
      destroy(total_start);
      destroy(total_stop);
      destroy(build_leaves_start);
      destroy(build_leaves_stop);
      destroy(fill_indices_start);
      destroy(fill_indices_stop);
      destroy(morton_start);
      destroy(morton_stop);
      destroy(sort_start);
      destroy(sort_stop);
      destroy(binary_build_start);
      destroy(binary_build_stop);
      destroy(conversion_start);
      destroy(conversion_stop);
      destroy(rt_start);
      destroy(rt_stop);
    }

    StepEvents(const StepEvents&) = delete;
    StepEvents& operator=(const StepEvents&) = delete;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t build_leaves_start = nullptr;
    cudaEvent_t build_leaves_stop = nullptr;
    cudaEvent_t fill_indices_start = nullptr;
    cudaEvent_t fill_indices_stop = nullptr;
    cudaEvent_t morton_start = nullptr;
    cudaEvent_t morton_stop = nullptr;
    cudaEvent_t sort_start = nullptr;
    cudaEvent_t sort_stop = nullptr;
    cudaEvent_t binary_build_start = nullptr;
    cudaEvent_t binary_build_stop = nullptr;
    cudaEvent_t conversion_start = nullptr;
    cudaEvent_t conversion_stop = nullptr;
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

template <unsigned int Arity>
RayTracingResult run_hploc_wide(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  const std::string experiment_name = "H-PLOC BVH" + std::to_string(Arity);

  std::cout << "\n=== Experiment: " << experiment_name << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_binary_nodes = 2 * n_faces - 1;
  size_t block_size = DEFAULT_GROUP_SIZE;
  const unsigned int binary_bvh_root_index = 2 * n_faces - 2;
  const unsigned long long root_task = cuda::hploc::pack_task(binary_bvh_root_index, 0);
  constexpr unsigned int one = 1;

  BVHNode* d_binary_bvh = nullptr;
  MortonCode* d_morton_codes = nullptr;
  MortonCode* d_morton_codes_sorted = nullptr;
  unsigned int* d_cluster_ids = nullptr;
  unsigned int* d_cluster_ids_sorted = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_n_binary_nodes = nullptr;

  WideBVHNode<Arity>* d_wide_bvh = nullptr;
  unsigned long long* d_tasks = nullptr;
  unsigned int* d_next_task = nullptr;
  unsigned int* d_next_wide_node = nullptr;
  void* d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_binary_bvh, sizeof(BVHNode) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes_sorted, sizeof(MortonCode) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids, sizeof(unsigned int) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids_sorted, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_binary_nodes, sizeof(unsigned int), stream));

  CUDA_SAFE_CALL(cudaMallocAsync(&d_wide_bvh, sizeof(WideBVHNode<Arity>) * max_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_tasks, sizeof(unsigned long long) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_next_task, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_next_wide_node, sizeof(unsigned int), stream));
  cub::DoubleBuffer<MortonCode> morton_codes_buffer(d_morton_codes, d_morton_codes_sorted);
  cub::DoubleBuffer<unsigned int> cluster_ids_buffer(d_cluster_ids, d_cluster_ids_sorted);
  cub::DeviceRadixSort::SortPairs(
      nullptr,
      sort_temp_storage_bytes,
      morton_codes_buffer,
      cluster_ids_buffer,
      static_cast<int>(n_faces),
      0,
      32,
      stream);
  CUDA_SAFE_CALL(cudaMallocAsync(&d_sort_temp_storage, sort_temp_storage_bytes, stream));
  CUDA_SYNC_STREAM(stream);

  std::vector<double> build_leaves_times;
  std::vector<double> fill_indices_times;
  std::vector<double> morton_times;
  std::vector<double> sort_times;
  std::vector<double> binary_build_times;
  std::vector<double> conversion_times;
  std::vector<double> total_build_times;
  std::vector<double> rt_times;
  std::vector<double> total_times;

  StepEvents events;

  const AdaptiveWarmupResult build_warmup = benchmark::run_adaptive([&](bool collect) {
    CUDA_SAFE_CALL(cudaEventRecord(events.total_start, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.build_leaves_start, stream));
    cuda::hploc::build_leaves_nodes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        scene.d_faces,
        n_faces,
        scene.d_vertices,
        d_binary_bvh);
    CUDA_SAFE_CALL(cudaEventRecord(events.build_leaves_stop, stream));

    auto cluster_ids = thrust::device_pointer_cast(d_cluster_ids);
    CUDA_SAFE_CALL(cudaEventRecord(events.fill_indices_start, stream));
    thrust::sequence(thrust::cuda::par_nosync.on(stream), cluster_ids, cluster_ids + n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.fill_indices_stop, stream));

    CUDA_SAFE_CALL(cudaMemsetAsync(d_parents, INVALID_INDEX, sizeof(unsigned int) * max_binary_nodes, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_n_binary_nodes, &n_faces, sizeof(n_faces), cudaMemcpyHostToDevice, stream));

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
    if (morton_codes_buffer.Current() != d_morton_codes) {
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_morton_codes, morton_codes_buffer.Current(), sizeof(MortonCode) * n_faces, cudaMemcpyDeviceToDevice, stream));
    }
    if (cluster_ids_buffer.Current() != d_cluster_ids) {
      CUDA_SAFE_CALL(cudaMemcpyAsync(d_cluster_ids, cluster_ids_buffer.Current(), sizeof(unsigned int) * n_faces, cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_SAFE_CALL(cudaEventRecord(events.sort_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.binary_build_start, stream));
    block_size = 128;
    cuda::hploc::build_kernel<<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(
        d_parents,
        d_morton_codes,
        d_binary_bvh,
        d_cluster_ids,
        d_n_binary_nodes,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.binary_build_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.conversion_start, stream));
    block_size = 128;
    CUDA_SAFE_CALL(cudaMemsetAsync(d_tasks, 0xFF, sizeof(unsigned long long) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_tasks, &root_task, sizeof(root_task), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_task, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_next_wide_node, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    cuda::hploc::convert_to_wide_kernel<<<div_ceil(n_faces, block_size), block_size, 0, stream>>>(
        d_binary_bvh,
        d_wide_bvh,
        reinterpret_cast<unsigned long long*>(d_tasks),
        d_next_task,
        d_next_wide_node,
        n_faces);
    CUDA_SAFE_CALL(cudaEventRecord(events.conversion_stop, stream));

    fb.clear();

    CUDA_SAFE_CALL(cudaEventRecord(events.rt_start, stream));
    cuda::hploc::rt_hploc_wide_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        d_wide_bvh,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera);
    CUDA_SAFE_CALL(cudaEventRecord(events.rt_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.total_stop, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(events.total_stop));

    const double build_leaves_ms = elapsed_ms(events.build_leaves_start, events.build_leaves_stop);
    const double fill_indices_ms = elapsed_ms(events.fill_indices_start, events.fill_indices_stop);
    const double morton_ms = elapsed_ms(events.morton_start, events.morton_stop);
    const double sort_ms = elapsed_ms(events.sort_start, events.sort_stop);
    const double binary_build_ms = elapsed_ms(events.binary_build_start, events.binary_build_stop);
    const double conversion_ms = elapsed_ms(events.conversion_start, events.conversion_stop);
    const double total_build_ms = build_leaves_ms + fill_indices_ms + morton_ms + sort_ms + binary_build_ms + conversion_ms;
    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      build_leaves_times.push_back(build_leaves_ms);
      fill_indices_times.push_back(fill_indices_ms);
      morton_times.push_back(morton_ms);
      sort_times.push_back(sort_ms);
      binary_build_times.push_back(binary_build_ms);
      conversion_times.push_back(conversion_ms);
      total_build_times.push_back(total_build_ms);
      rt_times.push_back(rt_ms);
      total_times.push_back(total_ms);
    }

    return total_ms;
  });
  print_warmup_report(experiment_name, build_warmup);

  const double build_mtris = n_faces * 1e-3 / stats::median(total_build_times);
  std::cout << experiment_name << " build leaves times (in ms) - " << stats::median(build_leaves_times) << std::endl;
  std::cout << experiment_name << " fill indices times (in ms) - " << stats::median(fill_indices_times) << std::endl;
  std::cout << experiment_name << " compute morton codes times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << experiment_name << " sort by key times (in ms) - " << stats::median(sort_times) << std::endl;
  std::cout << experiment_name << " build kernel times (in ms) - " << stats::median(binary_build_times) << std::endl;
  std::cout << experiment_name << " conversion times (in ms) - " << stats::median(conversion_times) << std::endl;
  std::cout << experiment_name << " total build times (in ms) - " << stats::median(total_build_times) << std::endl;
  std::cout << experiment_name << " total build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, d_next_wide_node, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_wide_nodes && n_wide_nodes <= max_binary_nodes, 63123333);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / stats::median(rt_times);
  std::cout << experiment_name << " ray tracing frame render times (in ms) - " << stats::median(rt_times) << std::endl;
  std::cout << experiment_name << " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << experiment_name << " total pipeline times (in ms) - " << stats::median(total_times) << std::endl;

  RayTracingResult res;
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" + experiment_name, res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_binary_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids_sorted, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_binary_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_wide_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_tasks, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_next_task, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_next_wide_node, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_sort_temp_storage, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}

template RayTracingResult run_hploc_wide<4>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
template RayTracingResult run_hploc_wide<8>(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir);
