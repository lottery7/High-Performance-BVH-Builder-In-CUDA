#include <cuda_runtime_api.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libimages/debug_io.h>

#include <filesystem>

#include "cpu_helpers/build_bvh_cpu.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include "utils/utils.h"

// ─────────────────────────────────────────────
// Данные сцены на GPU (RAII-обёртка)
// ─────────────────────────────────────────────

struct SceneGPU {
  float* vertices = nullptr;
  unsigned int* faces = nullptr;
  CameraViewGPU* camera = nullptr;

  unsigned int nvertices = 0;
  unsigned int nfaces = 0;

  SceneGPU(const SceneGeometry& scene, const CameraViewGPU& cam) : nvertices(scene.vertices.size()), nfaces(scene.faces.size())
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&vertices), 3 * nvertices * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&faces), 3 * nfaces * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&camera), sizeof(CameraViewGPU)));

    CUDA_SAFE_CALL(cudaMemcpy(vertices, scene.vertices.data(), 3 * nvertices * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(faces, scene.faces.data(), 3 * nfaces * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(camera, &cam, sizeof(CameraViewGPU), cudaMemcpyHostToDevice));
  }

  ~SceneGPU()
  {
    cudaFree(vertices);
    cudaFree(faces);
    cudaFree(camera);
  }
};

// ─────────────────────────────────────────────
// Фреймбуферы на GPU (RAII-обёртка)
// ─────────────────────────────────────────────

struct FramebuffersGPU {
  int* face_id = nullptr;
  float* ao = nullptr;

  unsigned int width = 0;
  unsigned int height = 0;

  FramebuffersGPU(unsigned int w, unsigned int h) : width(w), height(h)
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&face_id), w * h * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&ao), w * h * sizeof(float)));
  }

  void clear(cudaStream_t stream) const
  {
    cuda::fill(stream, face_id, NO_FACE_ID, width * height);
    cuda::fill(stream, ao, NO_AMBIENT_OCCLUSION, width * height);
    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
  }

  void readback(image32i& out_face_ids, image32f& out_ao) const
  {
    out_face_ids = image32i(width, height, 1);
    out_ao = image32f(width, height, 1);
    CUDA_SAFE_CALL(cudaMemcpy(out_face_ids.ptr(), face_id, width * height * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(out_ao.ptr(), ao, width * height * sizeof(float), cudaMemcpyDeviceToHost));
  }

  ~FramebuffersGPU()
  {
    cudaFree(face_id);
    cudaFree(ao);
  }
};

// ─────────────────────────────────────────────
// LBVH на GPU (RAII-обёртка)
// ─────────────────────────────────────────────

struct LBVHDataGPU {
  BVHNodeGPU* nodes = nullptr;
  unsigned int* leaf_face_indices = nullptr;

  LBVHDataGPU(const std::vector<BVHNodeGPU>& nodes_cpu, const std::vector<uint32_t>& indices_cpu)
  {
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&nodes), nodes_cpu.size() * sizeof(BVHNodeGPU)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&leaf_face_indices), indices_cpu.size() * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemcpy(nodes, nodes_cpu.data(), nodes_cpu.size() * sizeof(BVHNodeGPU), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(leaf_face_indices, indices_cpu.data(), indices_cpu.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  }

  ~LBVHDataGPU()
  {
    cudaFree(nodes);
    cudaFree(leaf_face_indices);
  }
};

// ─────────────────────────────────────────────
// Утилиты сохранения и проверки
// ─────────────────────────────────────────────

static void saveFramebuffers(const std::string& results_dir, const std::string& suffix, const image32i& face_ids, const image32f& ao)
{
  debug_io::dumpImage(results_dir + "/framebuffer_face_ids_" + suffix + ".bmp", debug_io::randomMapping(face_ids, NO_FACE_ID));
  debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_" + suffix + ".bmp", debug_io::depthMapping(ao));
}

static void validateAgainstBruteForce(
    const image32f& bf_ao,
    const image32i& bf_face_ids,
    const image32f& cmp_ao,
    const image32i& cmp_face_ids,
    unsigned int width,
    unsigned int height,
    uint64_t ao_error_code,
    uint64_t face_error_code)
{
  unsigned int ao_errors = countDiffs(bf_ao, cmp_ao, 0.01f);
  unsigned int face_errors = countDiffs(bf_face_ids, cmp_face_ids, 1);
  rassert(ao_errors < width * height / 100, ao_error_code, ao_errors, to_percent(ao_errors, width * height));
  rassert(face_errors < width * height / 100, face_error_code, face_errors, to_percent(face_errors, width * height));
}

// ─────────────────────────────────────────────
// Этап: Brute Force на GPU
// ─────────────────────────────────────────────

struct BruteForceResult {
  image32i face_ids;
  image32f ao;
  double total_time = 0.0;
};

static BruteForceResult runBruteForce(
    cudaStream_t stream,
    const SceneGPU& scene_gpu,
    const FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters)
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

// ─────────────────────────────────────────────
// Этап: RT с CPU-построенным LBVH
// ─────────────────────────────────────────────

struct CPULBVHResult {
  image32i face_ids;
  image32f ao;
  double lbvh_build_time = 0.0;
  double rt_time_sum = 0.0;
};

static CPULBVHResult runCPULBVH(
    cudaStream_t stream,
    const SceneGeometry& scene,
    const SceneGPU& scene_gpu,
    FramebuffersGPU& fb,
    const std::string& results_dir,
    int niters,
    std::vector<double>& out_perf_mrays)
{
  const unsigned int width = fb.width;
  const unsigned int height = fb.height;

  std::vector<BVHNodeGPU> nodes_cpu;
  std::vector<uint32_t> indices_cpu;

  timer cpu_lbvh_t;
  buildLBVH_CPU(scene.vertices, scene.faces, nodes_cpu, indices_cpu);
  double build_time = cpu_lbvh_t.elapsed();

  std::cout << "CPU build LBVH in " << build_time << " sec" << std::endl;
  std::cout << "CPU LBVH build performance: " << scene_gpu.nfaces * 1e-6f / build_time << " MTris/s" << std::endl;

  LBVHDataGPU lbvh(nodes_cpu, indices_cpu);

  fb.clear(stream);

  std::vector<double> rt_times;
  for (int iter = 0; iter < niters; ++iter) {
    timer t;
    cuda::ray_tracing_render_using_lbvh(
        stream,
        dim3(divCeil(width, 16), divCeil(height, 16)),
        dim3(16, 16),
        scene_gpu.vertices,
        scene_gpu.faces,
        lbvh.nodes,
        lbvh.leaf_face_indices,
        fb.face_id,
        fb.ao,
        scene_gpu.camera,
        scene_gpu.nfaces);
    rt_times.push_back(t.elapsed());
  }

  double mrays = width * height * AO_SAMPLES * 1e-6f / stats::median(rt_times);
  std::cout << "GPU with CPU LBVH ray tracing frame render times (in seconds) - " << stats::valuesStatsLine(rt_times) << std::endl;
  std::cout << "GPU with CPU LBVH ray tracing performance: " << mrays << " MRays/s" << std::endl;
  out_perf_mrays.push_back(mrays);

  CPULBVHResult result;
  result.lbvh_build_time = build_time;
  result.rt_time_sum = stats::sum(rt_times);
  fb.readback(result.face_ids, result.ao);
  saveFramebuffers(results_dir, "with_cpu_lbvh", result.face_ids, result.ao);

  // LBVHDataGPU освобождается здесь автоматически (RAII)
  return result;
}

// ─────────────────────────────────────────────
// Этап: RT с GPU-построенным LBVH (TODO)
// ─────────────────────────────────────────────

static void runGPULBVH(
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

// ─────────────────────────────────────────────
// Обработка одной сцены
// ─────────────────────────────────────────────

static void processScene(
    const std::string& scene_path,
    cudaStream_t stream,
    int niters,
    std::vector<double>& gpu_rt_perf_mrays_per_sec,
    std::vector<double>& gpu_lbvh_perfs_mtris_per_sec)
{
  std::cout << "____________________________________________________________________________________________" << std::endl;
  timer total_t;

  if (!std::filesystem::exists(scene_path)) {
    std::cout << "Scene not found: " << scene_path << std::endl;
    return;
  }

  // Загрузка сцены и камеры
  std::cout << "Loading scene " << scene_path << "..." << std::endl;
  timer loading_t;

  SceneGeometry scene = loadScene(scene_path);
  rassert(!scene.vertices.empty(), 546345423523143);
  rassert(!scene.faces.empty(), 54362452342);

  std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
  std::string camera_path = "data/" + scene_name + "/camera.txt";
  std::string results_dir = "results/" + scene_name;

  std::filesystem::create_directory("results");
  std::filesystem::create_directory(results_dir);

  std::cout << "Loading camera " << camera_path << "..." << std::endl;
  CameraViewGPU camera = loadViewState(camera_path);

  double loading_data_time = loading_t.elapsed();
  double images_saving_time = 0.0;
  double pcie_reading_time = 0.0;

  const unsigned int width = camera.K.width;
  const unsigned int height = camera.K.height;

  std::cout << "Scene " << scene_name << " loaded: " << scene.vertices.size() << " vertices, " << scene.faces.size() << " faces in "
            << loading_data_time << " sec" << std::endl;
  std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;

  // Аллоцируем данные на GPU — освобождаются автоматически при выходе из scope
  timer pcie_write_t;
  SceneGPU scene_gpu(scene, camera);
  FramebuffersGPU fb(width, height);
  double pcie_writing_time = pcie_write_t.elapsed();

  double cleaning_framebuffers_time = 0.0;
  double brute_force_total_time = 0.0;

  const bool has_brute_force = (scene.faces.size() < 1000);
  image32i bf_face_ids;
  image32f bf_ao;

  // Brute force
  if (has_brute_force) {
    timer save_t;
    auto bf = runBruteForce(stream, scene_gpu, fb, results_dir, niters);
    brute_force_total_time = bf.total_time;
    bf_face_ids = bf.face_ids;
    bf_ao = bf.ao;
    images_saving_time += save_t.elapsed();
  }

  // CPU LBVH
  double cpu_lbvh_time = 0.0;
  double rt_times_cpu_lbvh_sum = 0.0;
  {
    timer save_t;
    auto res = runCPULBVH(stream, scene, scene_gpu, fb, results_dir, niters, gpu_rt_perf_mrays_per_sec);
    cpu_lbvh_time = res.lbvh_build_time;
    rt_times_cpu_lbvh_sum = res.rt_time_sum;
    images_saving_time += save_t.elapsed();

    if (has_brute_force) {
      validateAgainstBruteForce(bf_ao, bf_face_ids, res.ao, res.face_ids, width, height, 345341512354123ULL, 3453415123546587ULL);
    }
  }

  // GPU LBVH (TODO)
  double gpu_lbvh_build_sum = 0.0;
  double gpu_lbvh_rt_sum = 0.0;
  bool gpu_lbvh_done = false;

  if (gpu_lbvh_done) {
    runGPULBVH(
        stream,
        scene_gpu,
        fb,
        results_dir,
        niters,
        gpu_rt_perf_mrays_per_sec,
        gpu_lbvh_perfs_mtris_per_sec,
        gpu_lbvh_build_sum,
        gpu_lbvh_rt_sum,
        bf_ao,
        bf_face_ids,
        has_brute_force);
  }

  // Итоговая статистика
  double total_time = total_t.elapsed();
  std::cout << "Scene processed in " << total_time << " sec = " << to_percent(loading_data_time, total_time) << " scene IO + ";
  if (has_brute_force) std::cout << to_percent(brute_force_total_time, total_time) << " brute force RT + ";
  std::cout << to_percent(cpu_lbvh_time, total_time) << " CPU LBVH + " << to_percent(rt_times_cpu_lbvh_sum, total_time) << " GPU with CPU LBVH + "
            << to_percent(gpu_lbvh_build_sum, total_time) << " GPU LBVH + " << to_percent(gpu_lbvh_rt_sum, total_time) << " GPU with GPU LBVH + "
            << to_percent(images_saving_time, total_time) << " images IO + " << to_percent(pcie_writing_time, total_time) << " PCI-E write + "
            << to_percent(pcie_reading_time, total_time) << " PCI-E read + " << to_percent(cleaning_framebuffers_time, total_time) << " cleaning VRAM"
            << std::endl;

  // SceneGPU и FramebuffersGPU освобождают GPU-память автоматически здесь
}

// ─────────────────────────────────────────────
// Точка входа
// ─────────────────────────────────────────────

void run(int argc, char** argv)
{
  cuda::selectCudaDevice(argc, argv);
  cudaStream_t stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  std::vector<std::string> scenes = {
      "data/gnome/gnome.ply",
      "data/powerplant/powerplant.obj",
      "data/san-miguel/san-miguel.obj",
  };

  const int niters = 10;
  std::vector<double> gpu_rt_perf_mrays_per_sec;
  std::vector<double> gpu_lbvh_perfs_mtris_per_sec;

  std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;

  for (const std::string& scene_path : scenes) processScene(scene_path, stream, niters, gpu_rt_perf_mrays_per_sec, gpu_lbvh_perfs_mtris_per_sec);

  CUDA_SAFE_CALL(cudaStreamDestroy(stream));

  std::cout << "____________________________________________________________________________________________" << std::endl;
  double avg_rt = stats::avg(gpu_rt_perf_mrays_per_sec);
  double avg_build = stats::avg(gpu_lbvh_perfs_mtris_per_sec);
  std::cout << "Total GPU RT with  LBVH avg perf: " << avg_rt << " MRays/sec (all " << stats::vectorToString(gpu_rt_perf_mrays_per_sec) << ")"
            << std::endl;
  std::cout << "Total building GPU LBVH avg perf: " << avg_build << " MTris/sec (all " << stats::vectorToString(gpu_lbvh_perfs_mtris_per_sec) << ")"
            << std::endl;
  std::cout << "Final score: " << avg_rt * avg_build << " coolness" << std::endl;

  if (gpu_rt_perf_mrays_per_sec.size() != 6 || gpu_lbvh_perfs_mtris_per_sec.size() != 3) std::cout << "Results are incomplete!" << std::endl;
}

int main(int argc, char** argv)
{
  try {
    run(argc, argv);
  } catch (std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}