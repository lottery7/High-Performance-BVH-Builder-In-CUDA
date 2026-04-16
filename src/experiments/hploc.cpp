#include "hploc.h"

#include "../kernels/h_ploc/hploc.h"
#include "../kernels/structs/framebuffers.h"
#include "../kernels/structs/scene.h"
#include "../utils/defines.h"
#include "../utils/utils.h"

#define EXPERIMENT_NAME "H-PLOC"

RayTracingResult run_hploc(cudaStream_t stream, const cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& results_dir)
{
  std::cout << "\n=== Experiment: " EXPERIMENT_NAME << std::endl;

  const unsigned int width = fb.width;
  const unsigned int height = fb.height;
  const unsigned int n_faces = scene.n_faces;

  BVHNode* d_bvh = nullptr;
  unsigned int* d_morton_codes = nullptr;
  unsigned int* d_cluster_ids = nullptr;
  unsigned int* d_parents = nullptr;
  unsigned int* d_n_clusters = nullptr;

  CUDA_SAFE_CALL(cudaMallocAsync(&d_bvh, sizeof(BVHNode) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_morton_codes, sizeof(unsigned int) * n_faces, stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_cluster_ids, sizeof(unsigned int) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_parents, sizeof(unsigned int) * (2 * n_faces - 1), stream));
  CUDA_SAFE_CALL(cudaMallocAsync(&d_n_clusters, sizeof(unsigned int), stream));
  CUDA_SYNC_STREAM(stream);

  const auto build_times = experiment_stats::benchmark_samples([&] {
    cuda::hploc::build(stream, scene.aabb, scene.d_faces, scene.d_vertices, d_bvh, d_parents, d_morton_codes, d_cluster_ids, d_n_clusters, n_faces);
    CUDA_SYNC_STREAM(stream);
  });
  experiment_stats::print_phase_stats(EXPERIMENT_NAME " build", build_times, n_faces, "MTris/s");
  unsigned int n_nodes = INVALID_INDEX;
  CUDA_SAFE_CALL(cudaMemcpyAsync(&n_nodes, d_n_clusters, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  std::cout << "Total nodes: " << n_nodes << std::endl;
  curassert(0 < n_nodes && n_nodes < 2 * n_faces, 541056);
  report_sah_hploc(stream, d_bvh, n_nodes, n_faces);

  fb.clear();

  const auto rt_times = experiment_stats::benchmark_samples([&] {
    cuda::rt_hploc(stream, width, height, scene.d_vertices, scene.d_faces, d_bvh, fb.d_face_id, fb.d_ao, scene.d_camera, scene.n_faces);
    CUDA_SYNC_STREAM(stream);
  });
  experiment_stats::print_phase_stats(EXPERIMENT_NAME " ray tracing frame render", rt_times, width * height * AO_SAMPLES, "MRays/s");
  experiment_stats::print_total_time_stats(EXPERIMENT_NAME " total frame time", build_times, rt_times);

  auto res = RayTracingResult();
  fb.readback(res.face_ids, res.ao);
  save_framebuffers(results_dir, "with_" EXPERIMENT_NAME, res.face_ids, res.ao);

  CUDA_SAFE_CALL(cudaFreeAsync(d_bvh, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_morton_codes, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_cluster_ids, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_parents, stream));
  CUDA_SAFE_CALL(cudaFreeAsync(d_n_clusters, stream));
  CUDA_SYNC_STREAM(stream);

  return res;
}
