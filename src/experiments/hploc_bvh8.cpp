#include "hploc_bvh8.h"

#include "../kernels/h_ploc/hploc.h"
#include "../kernels/h_ploc/wide_bvh_converter.h"
#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "../kernels/structs/wide_bvh_node.h"
#include "../utils/defines.h"
#include "../utils/utils.h"

#define EXPERIMENT_NAME "H-PLOC BVH8"

RayTracingResult run_hploc_bvh8(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;
  const unsigned int n_bvh2_capacity = 2 * n_faces - 1;
  const unsigned int n_wide_capacity = (4u * n_faces + 5u) / 7u;

  BVHNode* d_bvh2 = nullptr;
  WideBVHNode* d_wide_bvh = nullptr;
  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_cluster_ids = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_n_bvh2_nodes = nullptr;
  unsigned int* d_n_wide_nodes = nullptr;
  unsigned int* d_work_counter = nullptr;
  unsigned int* d_work_alloc_counter = nullptr;
  uint64_t* d_work_items = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh2, sizeof(BVHNode) * n_bvh2_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_wide_bvh, sizeof(WideBVHNode) * n_wide_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids, sizeof(unsigned int) * n_bvh2_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * n_bvh2_capacity, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_bvh2_nodes, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_wide_nodes, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_alloc_counter, sizeof(unsigned int), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_work_items, sizeof(uint64_t) * n_faces, stream));
  CUDA_SYNC_STREAM(stream);

  std::vector<double> build_times;
  std::vector<double> convert_times;
  std::vector<double> total_build_times;
  unsigned int n_bvh2_nodes = INVALID_INDEX;
  unsigned int n_wide_nodes = INVALID_INDEX;

  for (int iter = 0; iter < BENCHMARK_ITERS + WARMUP_ITERS; ++iter) {
    timer bvh_build_t;
    cuda::hploc::build(
        stream,
        scene.aabb,
        scene.d_faces,
        scene.d_vertices,
        d_bvh2,
        d_parents,
        d_morton_codes,
        d_cluster_ids,
        d_n_bvh2_nodes,
        n_faces);
    CUDA_SYNC_STREAM(stream);
    const double binary_build_time = bvh_build_t.elapsed();

    CUDA_SAFE_CALL(cudaMemcpyAsync(&n_bvh2_nodes, d_n_bvh2_nodes, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);

    timer wide_convert_t;
    cuda::wide_hploc::convert(stream, d_bvh2, d_wide_bvh, d_n_wide_nodes, d_work_items, d_work_counter, d_work_alloc_counter, n_bvh2_nodes, n_faces);
    CUDA_SYNC_STREAM(stream);
    const double wide_convert_time = wide_convert_t.elapsed();

    CUDA_SAFE_CALL(cudaMemcpyAsync(&n_wide_nodes, d_n_wide_nodes, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CUDA_SYNC_STREAM(stream);

    if (iter >= WARMUP_ITERS) {
      build_times.push_back(binary_build_time);
      convert_times.push_back(wide_convert_time);
      total_build_times.push_back(binary_build_time + wide_convert_time);
    }
  }

  experiment_stats::print_time_stats("H-PLOC binary build", build_times);
  experiment_stats::print_time_stats(EXPERIMENT_NAME " conversion", convert_times);
  experiment_stats::print_phase_stats(EXPERIMENT_NAME " build+convert", total_build_times, n_faces, "MTris/s");
  std::cout << "Binary nodes: " << n_bvh2_nodes << ", wide nodes: " << n_wide_nodes << std::endl;
  report_sah_wide(stream, d_wide_bvh, n_wide_nodes);

  fb.clear();

  const auto rt_times = experiment_stats::benchmark_samples([&] {
    cuda::rt_hploc_wide(stream, width, height, scene.d_vertices, scene.d_faces, d_wide_bvh, fb.d_face_id, fb.d_ao, scene.d_camera);
    CUDA_SYNC_STREAM(stream);
  });
  experiment_stats::print_phase_stats(EXPERIMENT_NAME " ray tracing frame render", rt_times, width * height * AO_SAMPLES, "MRays/s");
  experiment_stats::print_total_time_stats(EXPERIMENT_NAME " total frame time", total_build_times, rt_times);

  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_hploc_bvh8", res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh2, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_wide_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_bvh2_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_wide_nodes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_alloc_counter, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_work_items, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}
