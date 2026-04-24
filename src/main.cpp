#include <cuda_runtime_api.h>
#include <libbase/timer.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "experiments/common.h"
#include "experiments/hploc.h"
#include "experiments/hploc_wide.h"
#include "experiments/lbvh.h"
#include "experiments/nexus_bvh.h"
#include "experiments/nexus_bvh_wide.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/structs/framebuffers.h"
#include "kernels/structs/scene.h"
#include "utils/defines.h"
#include "utils/utils.h"

namespace
{
  constexpr const char* usage_text =
      "Usage: run_experiments --bench_iters <int> --experiments <list> --scenes <list> [--disable_warmup] [--device <int>]\n"
      "  experiments: lbvh, hploc, hploc_bvh4, hploc_bvh8, nexus_bvh, nexus_bvh_wide\n"
      "  examples:\n"
      "    run_experiments --bench_iters 10 --experiments hploc,hploc_bvh4,lbvh,nexus_bvh,nexus_bvh_wide --scenes "
      "data/gnome/gnome.ply,data/powerplant/powerplant.obj\n"
      "    run_experiments --bench_iters 5 --disable_warmup --experiments hploc_bvh4 --scenes data/hairball/hairball.obj --device 0";

  [[noreturn]] void throw_usage(const std::string& message) { throw std::runtime_error(message + "\n" + usage_text); }

  std::string normalize_experiment_name(std::string name)
  {
    name = trimmed(tolower(name));
    if (name == "hploc_wide4") return "hploc_bvh4";
    if (name == "hploc_wide8") return "hploc_bvh8";
    if (name == "nexus_bvh_wide8") return "nexus_bvh_wide";
    return name;
  }

  std::vector<std::string> parse_list(const std::string& value)
  {
    std::vector<std::string> values;
    for (const std::string& part : split(value, ",", false)) {
      const std::string item = trimmed(part);
      if (!item.empty()) values.push_back(item);
    }
    return values;
  }

  int parse_non_negative_int(const std::string& name, const std::string& value)
  {
    try {
      const int parsed = std::stoi(value);
      if (parsed < 0) throw_usage("Negative value for " + name + ": " + value);
      return parsed;
    } catch (const std::exception&) {
      throw_usage("Invalid value for " + name + ": " + value);
    }
  }

  int parse_positive_int(const std::string& name, const std::string& value)
  {
    const int parsed = parse_non_negative_int(name, value);
    if (parsed == 0) throw_usage("Zero value for " + name + ": " + value);
    return parsed;
  }

  bool is_option_name(std::string_view value) { return value.size() >= 2 && value[0] == '-' && value[1] == '-'; }

  std::string read_option_value(int argc, char** argv, int& i)
  {
    if (i + 1 >= argc || is_option_name(argv[i + 1])) {
      throw_usage("Missing value for option " + std::string(argv[i]));
    }
    return argv[++i];
  }

  std::string read_list_option_value(int argc, char** argv, int& i)
  {
    std::string value;
    while (i + 1 < argc) {
      const std::string_view next = argv[i + 1];
      if (is_option_name(next)) break;
      if (!value.empty()) value += ' ';
      value += argv[++i];
    }
    if (value.empty()) {
      throw_usage("Missing value for option " + std::string(argv[i]));
    }
    return value;
  }

  bool has_experiment(const RuntimeConfig& config, const std::string& experiment_name)
  {
    return std::find(config.experiments.begin(), config.experiments.end(), experiment_name) != config.experiments.end();
  }

  template <typename F>
  void run_experiment_if_enabled(
      const RuntimeConfig& config,
      const std::string& experiment_name,
      std::optional<RayTracingResult>& ground_truth,
      unsigned int width,
      unsigned int height,
      F&& run_experiment)
  {
    if (!has_experiment(config, experiment_name)) return;

    auto result = run_experiment();
    if (ground_truth)
      validate_against_ground_truth(*ground_truth, result, width, height);
    else
      ground_truth = std::move(result);
  }

  RuntimeConfig parse_runtime_config(int argc, char** argv)
  {
    if (argc <= 1) throw_usage("Missing command line arguments");

    RuntimeConfig config;
    bool has_benchmark_iters = false;
    bool has_experiments = false;
    bool has_scenes = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        throw_usage("Help requested");
      } else if (arg == "--bench_iters") {
        config.benchmark_iters = parse_positive_int(arg, read_option_value(argc, argv, i));
        has_benchmark_iters = true;
      } else if (arg == "--disable_warmup") {
        config.disable_warmup = true;
      } else if (arg == "--device") {
        config.cuda_device = parse_non_negative_int(arg, read_option_value(argc, argv, i));
      } else if (arg == "--experiments") {
        config.experiments = parse_list(read_list_option_value(argc, argv, i));
        for (std::string& experiment_name : config.experiments) {
          experiment_name = normalize_experiment_name(experiment_name);
          const bool valid = experiment_name == "lbvh" || experiment_name == "hploc" || experiment_name == "hploc_bvh4" ||
                             experiment_name == "hploc_bvh8" || experiment_name == "nexus_bvh" || experiment_name == "nexus_bvh_wide";
          if (!valid) throw_usage("Unknown experiment: " + experiment_name);
        }
        has_experiments = !config.experiments.empty();
      } else if (arg == "--scenes") {
        config.scenes = parse_list(read_list_option_value(argc, argv, i));
        has_scenes = !config.scenes.empty();
      } else {
        throw_usage("Unknown option: " + arg);
      }
    }

    if (!has_benchmark_iters || !has_experiments || !has_scenes) {
      throw_usage("Missing required options");
    }

    return config;
  }
}  // namespace

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

  const RuntimeConfig& config = runtime_config_const();
  std::optional<RayTracingResult> ground_truth;

  run_experiment_if_enabled(config, "lbvh", ground_truth, width, height, [&] { return run_lbvh(stream, scene_gpu, fb, results_dir); });

  run_experiment_if_enabled(config, "hploc", ground_truth, width, height, [&] { return run_hploc(stream, scene_gpu, fb, results_dir); });

  run_experiment_if_enabled(config, "hploc_bvh4", ground_truth, width, height, [&] { return run_hploc_wide<4>(stream, scene_gpu, fb, results_dir); });

  run_experiment_if_enabled(config, "hploc_bvh8", ground_truth, width, height, [&] { return run_hploc_wide<8>(stream, scene_gpu, fb, results_dir); });

  run_experiment_if_enabled(config, "nexus_bvh", ground_truth, width, height, [&] { return run_nexus_bvh(stream, scene_gpu, fb, results_dir); });

  run_experiment_if_enabled(config, "nexus_bvh_wide", ground_truth, width, height, [&] { return run_nexus_bvh_wide(stream, scene_gpu, fb, results_dir); });
}

static void run(int argc, char** argv)
{
  if (RASSERT_ENABLED) {
    std::cout << "CUDA rassert enabled. It will impact performance." << std::endl;
  }
  runtime_config() = parse_runtime_config(argc, argv);
  cuda::select_cuda_device(runtime_config_const().cuda_device);
  cudaStream_t stream;
  CUDA_SAFE_CALL(cudaStreamCreate(&stream));

  std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion" << std::endl;

  for (const std::string& scene_path : runtime_config_const().scenes) {
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
