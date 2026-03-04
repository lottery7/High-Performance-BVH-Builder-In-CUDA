#include <cuda_runtime_api.h>
#include <libbase/timer.h>

#include <filesystem>
#include <optional>

#include "experiments/common.h"
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

static void processScene(const std::string& scene_path, cudaStream_t stream, int niters)
{
  std::cout << "____________________________________________________________________________________________" << std::endl;

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

  const unsigned int width = camera.K.width;
  const unsigned int height = camera.K.height;

  SceneGPU scene_gpu(scene, camera);
  FramebuffersGPU fb(width, height);

  std::cout << "Scene " << scene_name << " loaded to GPU: " << scene.vertices.size() << " vertices, " << scene.faces.size() << " faces in "
            << loading_data_time << " sec" << std::endl;
  std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;

  // Brute force
  std::optional<RayTracingResult> bf_res;
  if (scene.faces.size() < 1000) {
    bf_res = runBruteForce(stream, scene_gpu, fb, results_dir, niters);
  }

  // CPU LBVH
  {
    auto res = runCPULBVH(stream, scene, scene_gpu, fb, results_dir, niters);
    if (bf_res) {
      validateAgainstGroundTruth(*bf_res, res, width, height);
    }
  }

  // GPU LBVH
  {
    auto res = runGPULBVH(stream, scene_gpu, fb, results_dir, niters);
    if (bf_res) {
      validateAgainstGroundTruth(*bf_res, res, width, height);
    }
  }
}

static void run(int argc, char** argv)
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

  std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;

  for (const std::string& scene_path : scenes) processScene(scene_path, stream, niters);

  CUDA_SAFE_CALL(cudaStreamDestroy(stream));
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