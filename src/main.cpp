#include <cuda_runtime_api.h>
#include <libbase/timer.h>

#include <filesystem>
#include <iostream>
#include <optional>

#include "experiments/common.h"
#include "experiments/hploc.h"
#include "experiments/hploc_wide.h"
#include "experiments/kitten_lbvh.h"
#include "experiments/my_cpu_lbvh.h"
#include "experiments/my_gpu_lbvh.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/structs/framebuffers.h"
#include "kernels/structs/scene.h"
#include "utils/defines.h"
#include "utils/utils.h"

static void process_scene(cudaStream_t stream, const std::string& scene_path)
{
  std::cout << "____________________________________________________________________________________________" << std::endl;

  std::cout << "Loading scene " << scene_path << "..." << std::endl;
  timer loading_t;

  if (!std::filesystem::exists(scene_path)) {
    std::cout << "Scene not found: " << scene_path << std::endl;
    return;
  }

  SceneGeometry scene = load_scene(scene_path);
  rassert(!scene.vertices.empty(), 546345423523143);
  rassert(!scene.faces.empty(), 54362452342);

  std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
  std::string camera_path = "data/" + scene_name + "/camera.txt";
  std::string results_dir = "results/" + scene_name;

  std::filesystem::create_directory("results");
  std::filesystem::create_directory(results_dir);

  std::cout << "Loading camera " << camera_path << "..." << std::endl;
  CameraView camera = load_view_state(camera_path);

  const double loading_data_time = loading_t.elapsed();

  const unsigned int width = camera.K.width;
  const unsigned int height = camera.K.height;

  cuda::Scene scene_gpu(stream, scene, camera);
  cuda::Framebuffers fb(stream, width, height);
  CUDA_SYNC_STREAM(stream);

  std::cout << "Scene " << scene_name << " loaded to GPU: " << scene.vertices.size() << " vertices, " << scene.faces.size() << " faces in "
            << loading_data_time << " sec" << std::endl;
  std::cout << "Camera framebuffer size: " << width << "x" << height << std::endl;
  std::cout << "Running experiments" << std::endl << std::endl;

  std::optional<RayTracingResult> ground_truth;

  // CPU LBVH
  // {
  //   auto res = run_cpu_lbvh(stream, scene, scene_gpu, fb, results_dir);
  //   if (ground_truth)
  //     validate_against_ground_truth(*ground_truth, res, width, height);
  //   else
  //     ground_truth = res;
  // }

  // My implementation of LBVH
  // {
  //   auto res = run_my_gpu_lbvh(stream, scene_gpu, fb, results_dir);
  //   if (ground_truth)
  //     validate_against_ground_truth(*ground_truth, res, width, height);
  //   else
  //     ground_truth = res;
  // }

  // Kitten LBVH (works VERY bad on large scenes)
  // {
  //   auto res = run_kitten_lbvh(scene_gpu, fb, results_dir);
  //   if (ground_truth)
  //     validate_against_ground_truth(*ground_truth, res, width, height);
  //   else
  //     ground_truth = res;
  // }

  // My implementation of H-PLOC
  {
    auto res = run_hploc(stream, scene_gpu, fb, results_dir);
    if (ground_truth)
      validate_against_ground_truth(*ground_truth, res, width, height);
    else
      ground_truth = res;
  }

  // H-PLOC + conversion to BVH4
  {
    auto res = run_hploc_wide<4>(stream, scene_gpu, fb, results_dir);
    if (ground_truth)
      validate_against_ground_truth(*ground_truth, res, width, height);
    else
      ground_truth = res;
  }

  // H-PLOC + conversion to BVH8
  {
    auto res = run_hploc_wide<8>(stream, scene_gpu, fb, results_dir);
    if (ground_truth)
      validate_against_ground_truth(*ground_truth, res, width, height);
    else
      ground_truth = res;
  }
}

static void run(int argc, char** argv)
{
  cuda::select_cuda_device(argc, argv);
  cudaStream_t stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  std::vector<std::string> scenes;
  for (int i = 1; i < argc; ++i) {
    scenes.push_back(argv[i]);
  }

  if (scenes.empty()) {
    scenes = {
        // "data/gnome/gnome.ply",
        "data/hairball/hairball.obj",
        // "data/powerplant/powerplant.obj",
        // "data/san-miguel/san-miguel.obj",
    };
  }

  std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;

  for (const std::string& scene_path : scenes) {
    process_scene(stream, scene_path);
  }

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
