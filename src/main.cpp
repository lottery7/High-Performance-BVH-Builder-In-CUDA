#include <cuda_runtime_api.h>
#include <libbase/timer.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "experiments/common.h"
#include "experiments/hploc_bvh2.h"
#include "experiments/hploc_bvh8.h"
#include "experiments/lbvh.h"
#include "experiments/nexus_bvh2.h"
#include "experiments/nexus_bvh8.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/structs/framebuffers.h"
#include "kernels/structs/scene.h"
#include "utils/defines.h"
#include "utils/utils.h"

namespace
{
  using ExperimentRunner = std::function<RayTracingResult(cudaStream_t, cuda::Scene&, cuda::Framebuffers&, const std::string&)>;

  struct ExperimentSpec {
    std::string name;
    std::vector<std::string> aliases;
    ExperimentRunner run;
  };

  constexpr const char* usage_text =
      "Usage: run_experiments --bench_iters <int> --experiments <list> --scenes <list> [--disable_warmup] [--device <int>]\n"
      "  examples:\n"
      "    run_experiments --bench_iters 10 --experiments hploc,hploc_bvh4,lbvh,nexus_bvh2,nexus_bvh8 --scenes "
      "data/gnome/gnome.ply,data/powerplant/powerplant.obj\n"
      "    run_experiments --bench_iters 5 --disable_warmup --experiments hploc_bvh4 --scenes data/hairball/hairball.obj --device 0";

  [[noreturn]] void throw_usage(const std::string& message) { throw std::runtime_error(message + "\n" + usage_text); }

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
    while (i + 1 < argc && !is_option_name(argv[i + 1])) {
      if (!value.empty()) value += ' ';
      value += argv[++i];
    }
    if (value.empty()) {
      throw_usage("Missing value for option " + std::string(argv[i]));
    }
    return value;
  }

  std::vector<std::string> parse_list(const std::string& value)
  {
    std::vector<std::string> out;
    for (const auto& part : split(value, ",", false)) {
      std::string s = trimmed(part);
      if (!s.empty()) out.push_back(s);
    }
    return out;
  }

  int parse_non_negative_int(const std::string& name, const std::string& value)
  {
    try {
      int parsed = std::stoi(value);
      if (parsed < 0) throw_usage("Negative value for " + name + ": " + value);
      return parsed;
    } catch (...) {
      throw_usage("Invalid value for " + name + ": " + value);
    }
  }

  int parse_positive_int(const std::string& name, const std::string& value)
  {
    int parsed = parse_non_negative_int(name, value);
    if (parsed == 0) throw_usage("Zero value for " + name + ": " + value);
    return parsed;
  }

  const std::vector<ExperimentSpec>& experiments_registry()
  {
    static const std::vector<ExperimentSpec> experiments = {
        {
            "lbvh",
            {"lbvh"},
            [](cudaStream_t s, cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& dir) { return run_lbvh(s, scene, fb, dir); },
        },
        {
            "hploc",
            {"hploc_bvh2"},
            [](cudaStream_t s, cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& dir) { return run_hploc(s, scene, fb, dir); },
        },
        {
            "hploc_bvh8",
            {"hploc_bvh8"},
            [](cudaStream_t s, cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& dir) { return run_hploc_bvh8(s, scene, fb, dir); },
        },
        {
            "nexus_bvh2",
            {"nexus_bvh2"},
            [](cudaStream_t s, cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& dir) { return run_nexus_bvh2(s, scene, fb, dir); },
        },
        {
            "nexus_bvh8",
            {"nexus_bvh8"},
            [](cudaStream_t s, cuda::Scene& scene, cuda::Framebuffers& fb, const std::string& dir) { return run_nexus_bvh8(s, scene, fb, dir); },
        },
    };
    return experiments;
  }

  const std::unordered_map<std::string, const ExperimentSpec*>& experiments_by_name()
  {
    static const std::unordered_map<std::string, const ExperimentSpec*> map = [] {
      std::unordered_map<std::string, const ExperimentSpec*> m;
      for (const auto& e : experiments_registry()) {
        m[e.name] = &e;
        for (const auto& alias : e.aliases) m[alias] = &e;
      }
      return m;
    }();
    return map;
  }

  std::string normalize_experiment_name(std::string name)
  {
    name = trimmed(tolower(name));
    const auto& map = experiments_by_name();
    auto it = map.find(name);
    if (it == map.end()) throw_usage("Unknown experiment: " + name);
    return it->second->name;
  }

  RuntimeConfig parse_runtime_config(int argc, char** argv)
  {
    if (argc <= 1) throw_usage("Missing command line arguments");

    RuntimeConfig config;
    bool has_benchmark_iters = false;
    bool has_experiments = false;
    bool has_scenes = false;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

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
        for (auto& name : config.experiments) name = normalize_experiment_name(name);

        // убираем дубли, сохраняя порядок
        std::unordered_set<std::string> seen;
        std::vector<std::string> unique;
        for (const auto& name : config.experiments) {
          if (seen.insert(name).second) unique.push_back(name);
        }
        config.experiments = std::move(unique);
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

  void run_selected_experiments(
      cudaStream_t stream,
      cuda::Scene& scene_gpu,
      cuda::Framebuffers& fb,
      const std::string& results_dir,
      unsigned int width,
      unsigned int height)
  {
    std::optional<RayTracingResult> ground_truth;
    const auto& registry = experiments_registry();
    const auto& enabled = runtime_config_const().experiments;

    for (const auto& selected_name : enabled) {
      auto it = std::find_if(registry.begin(), registry.end(), [&](const ExperimentSpec& e) { return e.name == selected_name; });
      if (it == registry.end()) continue;  // не должно происходить после валидации

      auto result = it->run(stream, scene_gpu, fb, results_dir);
      if (ground_truth)
        validate_against_ground_truth(*ground_truth, result, width, height);
      else
        ground_truth = std::move(result);
    }
  }

  void process_scene(cudaStream_t stream, const std::string& scene_path)
  {
    std::cout << "____________________________________________________________________________________________\n";
    std::cout << "Loading scene " << scene_path << "...\n";

    timer loading_t;

    if (!std::filesystem::exists(scene_path)) {
      std::cout << "Scene not found: " << scene_path << '\n';
      return;
    }

    SceneGeometry scene = load_scene(scene_path);
    rassert(!scene.vertices.empty(), 546345423523143);
    rassert(!scene.faces.empty(), 54362452342);

    const std::string scene_name = std::filesystem::path(scene_path).parent_path().filename().string();
    const std::string camera_path = "data/" + scene_name + "/camera.txt";
    const std::string results_dir = "results/" + scene_name;

    std::filesystem::create_directory("results");
    std::filesystem::create_directory(results_dir);

    std::cout << "Loading camera " << camera_path << "...\n";
    CameraView camera = load_view_state(camera_path);

    const double loading_data_time = loading_t.elapsed();
    const unsigned int width = camera.K.width;
    const unsigned int height = camera.K.height;

    cuda::Scene scene_gpu(stream, scene, camera);
    cuda::Framebuffers fb(stream, width, height);
    CUDA_SYNC_STREAM(stream);

    std::cout << "Scene " << scene_name << " loaded to GPU: " << scene.vertices.size() << " vertices, " << scene.faces.size() << " faces in "
              << loading_data_time << " sec\n";
    std::cout << "Camera framebuffer size: " << width << "x" << height << '\n';
    std::cout << "Running experiments\n\n";

    run_selected_experiments(stream, scene_gpu, fb, results_dir, width, height);
  }

  void run(int argc, char** argv)
  {
    if (RASSERT_ENABLED) {
      std::cout << "CUDA rassert enabled. It will impact performance." << std::endl;
    }

    runtime_config() = parse_runtime_config(argc, argv);
    cuda::select_cuda_device(runtime_config_const().cuda_device);

    cudaStream_t stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    std::cout << "Using " << AO_SAMPLES << " ray samples for ambient occlusion\n";

    for (const auto& scene_path : runtime_config_const().scenes) {
      process_scene(stream, scene_path);
    }

    CUDA_SAFE_CALL(cudaStreamDestroy(stream));
  }
}  // namespace

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
