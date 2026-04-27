#include <libbase/stats.h>

#include <cfloat>
#include <cub/device/device_radix_sort.cuh>
#include <vector>

#include "../kernels/helpers/helpers.cuh"
#include "../kernels/nexus_bvh/nexus_bvh2.cuh"
#include "../kernels/nexus_bvh/nexus_bvh8.cuh"
#include "../utils/defines.h"
#include "../utils/utils.h"
#include "../utils/wide_bvh_sah.h"
#include "benchmark.h"
#include "kernels/ray_tracing/rt_bvh8.cuh"
#include "kernels/ray_tracing/rt_compressed_bvh8.cuh"
#include "nexus_bvh8.h"

#define EXPERIMENT_NAME "NexusBVH BVH8"

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
      destroy(scene_bounds_start);
      destroy(scene_bounds_stop);
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
    cudaEvent_t scene_bounds_start = nullptr;
    cudaEvent_t scene_bounds_stop = nullptr;
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

RayTracingResult run_nexus_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int max_binary_nodes = 2 * n_faces - 1;
  const unsigned int max_wide_nodes = static_cast<unsigned int>(div_ceil(static_cast<int>(4u * n_faces - 1u), 7));

  BVH2Node* d_binary_bvh = nullptr;
  cuda::nexus_bvh::Workspace workspace;
  cuda::nexus_bvh_wide::BVH8Node* d_wide_bvh = nullptr;
  unsigned int* d_prim_idx = nullptr;
  unsigned int* d_node_counter = nullptr;
  unsigned int* d_leaf_counter = nullptr;
  unsigned int* d_work_counter = nullptr;
  unsigned int* d_work_alloc_counter = nullptr;
  std::uint64_t* d_index_pairs = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_binary_bvh, sizeof(BVH2Node) * max_binary_nodes, stream));
  cuda::nexus_bvh::allocate_workspace(stream, workspace, n_faces);
  CUDA_SAFE_CALL(cudaMallocAsync(&d_wide_bvh, sizeof(cuda::nexus_bvh_wide::BVH8Node) * max_wide_nodes, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_prim_idx, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_node_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_leaf_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_alloc_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_index_pairs, sizeof(std::uint64_t) * n_faces, stream));
  CUDA_SYNC_STREAM(stream);

  cuda::nexus_bvh::BuildState build_state{};
  build_state.scene_bounds = workspace.d_scene_bounds;
  build_state.nodes = d_binary_bvh;
  build_state.cluster_indices = workspace.d_cluster_indices;
  build_state.parent_indices = workspace.d_parent_indices;
  build_state.prim_count = n_faces;
  build_state.cluster_count = workspace.d_cluster_count;

  cuda::nexus_bvh_wide::BVH8BuildState wide_build_state{};
  wide_build_state.bvh2Nodes = reinterpret_cast<const cuda::nexus_bvh_wide::BVH2Node*>(d_binary_bvh);
  wide_build_state.bvh8Nodes = d_wide_bvh;
  wide_build_state.primIdx = d_prim_idx;
  wide_build_state.primCount = n_faces;
  wide_build_state.nodeCounter = d_node_counter;
  wide_build_state.leafCounter = d_leaf_counter;
  wide_build_state.indexPairs = d_index_pairs;
  wide_build_state.workCounter = d_work_counter;
  wide_build_state.workAllocCounter = d_work_alloc_counter;

  const AABB empty_scene_bounds{FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
  const std::uint64_t first_pair = static_cast<std::uint64_t>(max_binary_nodes - 1) << 32u;
  const unsigned int zero = 0;
  const unsigned int one = 1;

  std::vector<double> scene_bounds_times;
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

    build_state.cluster_indices = workspace.d_cluster_indices;
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.scene_bounds, &empty_scene_bounds, sizeof(AABB), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(build_state.parent_indices, 0xff, sizeof(unsigned int) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(build_state.cluster_count, &n_faces, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.scene_bounds_start, stream));
    cuda::nexus_bvh::compute_scene_bounds_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(
        build_state,
        scene.d_faces,
        scene.d_vertices);
    CUDA_SAFE_CALL(cudaEventRecord(events.scene_bounds_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.morton_start, stream));
    cuda::nexus_bvh::compute_morton_codes_kernel<<<compute_grid(n_faces), DEFAULT_GROUP_SIZE, 0, stream>>>(build_state, workspace.d_morton_codes);
    CUDA_SAFE_CALL(cudaEventRecord(events.morton_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.sort_start, stream));
    cuda::sort_pairs(
        stream,
        workspace.d_morton_codes,
        workspace.d_morton_codes_sorted,
        workspace.d_cluster_indices,
        workspace.d_cluster_indices_sorted,
        n_faces,
        0,
        32);
    CUDA_SAFE_CALL(cudaEventRecord(events.sort_stop, stream));

    build_state.cluster_indices = workspace.d_cluster_indices_sorted;
    constexpr int binary_block_size = 64;
    CUDA_SAFE_CALL(cudaEventRecord(events.binary_build_start, stream));
    cuda::nexus_bvh::build_bvh2_kernel<<<div_ceil(n_faces, binary_block_size), binary_block_size, 0, stream>>>(
        build_state,
        workspace.d_morton_codes_sorted);
    CUDA_SAFE_CALL(cudaEventRecord(events.binary_build_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.conversion_start, stream));
    CUDA_SAFE_CALL(cudaMemsetAsync(d_index_pairs, 0xFF, sizeof(std::uint64_t) * n_faces, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_index_pairs, &first_pair, sizeof(first_pair), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_node_counter, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_alloc_counter, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_work_counter, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));
    CUDA_SAFE_CALL(cudaMemcpyAsync(d_leaf_counter, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));
    constexpr int wide_block_size = 256;
    cuda::nexus_bvh_wide::build_bvh8_kernel<<<div_ceil(n_faces, wide_block_size), wide_block_size, 0, stream>>>(wide_build_state);
    CUDA_SAFE_CALL(cudaEventRecord(events.conversion_stop, stream));

    fb.clear();

    CUDA_SAFE_CALL(cudaEventRecord(events.rt_start, stream));
    cuda::rt_bvh8_kernel<<<compute_grid(width, height), DEFAULT_GROUP_SIZE_2D, 0, stream>>>(
        scene.d_vertices,
        scene.d_faces,
        reinterpret_cast<BVH8Node*>(d_wide_bvh),
        d_prim_idx,
        0,
        fb.d_face_id,
        fb.d_ao,
        scene.d_camera);
    CUDA_SAFE_CALL(cudaEventRecord(events.rt_stop, stream));

    CUDA_SAFE_CALL(cudaEventRecord(events.total_stop, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(events.total_stop));

    const double scene_bounds_ms = elapsed_ms(events.scene_bounds_start, events.scene_bounds_stop);
    const double morton_ms = elapsed_ms(events.morton_start, events.morton_stop);
    const double sort_ms = elapsed_ms(events.sort_start, events.sort_stop);
    const double binary_build_ms = elapsed_ms(events.binary_build_start, events.binary_build_stop);
    const double conversion_ms = elapsed_ms(events.conversion_start, events.conversion_stop);
    const double total_build_ms = scene_bounds_ms + morton_ms + sort_ms + binary_build_ms + conversion_ms;
    const double rt_ms = elapsed_ms(events.rt_start, events.rt_stop);
    const double total_ms = elapsed_ms(events.total_start, events.total_stop);

    if (collect) {
      scene_bounds_times.push_back(scene_bounds_ms);
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
  print_warmup_report(EXPERIMENT_NAME, build_warmup);

  const double build_mtris = n_faces * 1e-3 / stats::median(total_build_times);
  std::cout << EXPERIMENT_NAME " compute scene bounds times (in ms) - " << stats::median(scene_bounds_times) << std::endl;
  std::cout << EXPERIMENT_NAME " compute morton codes times (in ms) - " << stats::median(morton_times) << std::endl;
  std::cout << EXPERIMENT_NAME " radix sort times (in ms) - " << stats::median(sort_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build kernel times (in ms) - " << stats::median(binary_build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " conversion times (in ms) - " << stats::median(conversion_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build pipeline times (in ms) - " << stats::median(total_build_times) << std::endl;
  std::cout << EXPERIMENT_NAME " build performance: " << build_mtris << " MTris/s" << std::endl;

  unsigned int n_wide_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, d_node_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  curassert(0 < n_wide_nodes && n_wide_nodes <= max_wide_nodes, 226786315);
  wide_bvh_sah::report_nexus_bvh8_sah(stream, d_wide_bvh, n_wide_nodes);

  const double mrays = width * height * AO_SAMPLES * 1e-3 / stats::median(rt_times);
  std::cout << EXPERIMENT_NAME " ray tracing frame render times (in ms) - " << stats::median(rt_times) << std::endl;
  std::cout << EXPERIMENT_NAME " ray tracing performance: " << mrays << " MRays/s" << std::endl;
  std::cout << EXPERIMENT_NAME " total pipeline times (in ms) - " << stats::median(total_times) << std::endl;

  RayTracingResult result;
  fb.readback(result.face_ids, result.ao);
  save_framebuffers(results_dir, "with_nexus_bvh_wide", result.face_ids, result.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_index_pairs, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_alloc_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_leaf_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_node_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_prim_idx, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_wide_bvh, stream));
  cuda::nexus_bvh::free_workspace(stream, workspace);
  CUDA_SAFE_CALL(cudaFreeAsync(d_binary_bvh, stream));
  CUDA_SYNC_STREAM(stream);

  return result;
}
