#include <cuda_runtime_api.h>
#include <libbase/stats.h>
#include <libbase/timer.h>

#include <filesystem>

#include "experiments/cpu_lbvh.h"
#include "experiments/gpu_brute_force.h"
#include "experiments/gpu_lbvh.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "utils/cuda_utils.h"
#include "utils/gpu_wrappers.h"
#include "utils/utils.h"

static void processScene(
    const std::string& scene_path,
    cudaStream_t stream,
    int niters,
    std::vector<double>& gpu_rt_perf_mrays_per_sec,
    std::vector<double>& gpu_lbvh_perfs_mtris_per_sec)
{
  std::cout << "____________________________________________________________________________________________" << std::endl;
  timer total_t;

  std::cout << "Loading scene " << scene_path << "..." << std::endl;
  timer loading_t;

  if (!std::filesystem::exists(scene_path)) {
    std::cout << "Scene not found: " << scene_path << std::endl;
    return;
  }

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
            << to_percent(pcie_reading_time, total_time) << " PCI-E read" << std::endl;
}

static void run(int argc, char** argv)
{
  cuda::selectCudaDevice(argc, argv);
  cudaStream_t stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  std::vector<std::string> scenes = {
      "data/gnome/gnome.ply",
      // "data/powerplant/powerplant.obj",
      // "data/san-miguel/san-miguel.obj",
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
  std::cout << "Total GPU RT with LBVH avg perf: " << avg_rt << " MRays/sec (all " << stats::vectorToString(gpu_rt_perf_mrays_per_sec) << ")"
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