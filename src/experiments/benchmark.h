#pragma once

#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <libbase/stats.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../utils/utils.h"
#include "common.h"

struct AdaptiveWarmupResult {
  int iterations = 0;
  double total_ms = 0.0;
  bool stabilized = false;
  bool disabled = false;
};

inline void print_warmup_report(const std::string& label, const AdaptiveWarmupResult& warmup)
{
  if (warmup.disabled) {
    std::cout << label << " adaptive warmup: disabled" << std::endl;
    return;
  }

  const int warmup_limit_ms = runtime_config_const().warmup_max_seconds * 1000;
  std::cout << label << " adaptive warmup: " << warmup.iterations << " iterations in " << warmup.total_ms << " ms ("
            << (warmup.stabilized ? "stabilized" : "hit time limit") << ")" << std::endl;
  if (!warmup.stabilized) {
    std::cout << "WARNING: " << label << " warmup did not stabilize within " << warmup_limit_ms << " ms" << std::endl;
  }
}

class CudaEventTimer
{
 public:
  CudaEventTimer()
  {
    CUDA_SAFE_CALL(cudaEventCreate(&start_));
    CUDA_SAFE_CALL(cudaEventCreate(&stop_));
  }

  ~CudaEventTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;

  template <typename F>
  float measure(cudaStream_t stream, F&& fn)
  {
    CUDA_SAFE_CALL(cudaEventRecord(start_, stream));
    std::forward<F>(fn)();
    CUDA_SAFE_CALL(cudaEventRecord(stop_, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_));

    float elapsed_ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return elapsed_ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

namespace benchmark
{
  inline bool warmup_stabilized(const std::vector<double>& warmup_times_ms)
  {
    constexpr int window = 5;
    constexpr double drift_tolerance = 0.02;
    constexpr double spread_tolerance = 0.03;

    if (warmup_times_ms.size() < 2 * window) return false;

    const auto current_begin = warmup_times_ms.end() - window;
    const auto previous_begin = current_begin - window;
    std::vector<double> previous_window(previous_begin, current_begin);
    std::vector<double> current_window(current_begin, warmup_times_ms.end());

    const double previous_median = stats::median(previous_window);
    const double current_median = stats::median(current_window);
    const double scale = std::max(current_median, std::numeric_limits<double>::epsilon());
    const double drift = std::abs(current_median - previous_median) / scale;
    const double spread = (stats::percentile(current_window, 90) - stats::percentile(current_window, 10)) / scale;

    return drift <= drift_tolerance && spread <= spread_tolerance;
  }

  template <typename MeasureOnce>
  AdaptiveWarmupResult run_adaptive(MeasureOnce&& measure_once)
  {
    const RuntimeConfig& config = runtime_config_const();
    if (config.disable_warmup) {
      AdaptiveWarmupResult result;
      result.disabled = true;
      for (int iter = 0; iter < benchmark_iters(); ++iter) {
        measure_once(true);
      }
      return result;
    }

    std::vector<double> warmup_times;
    warmup_times.reserve(128);

    AdaptiveWarmupResult result;
    const double min_warmup_ms = static_cast<double>(config.warmup_min_seconds) * 1e3;
    const double max_warmup_ms = static_cast<double>(config.warmup_max_seconds) * 1e3;

    while (result.total_ms < max_warmup_ms) {
      const double sample_ms = measure_once(false);
      warmup_times.push_back(sample_ms);
      result.total_ms += sample_ms;
      ++result.iterations;

      const bool has_min_time = result.total_ms >= min_warmup_ms;
      if (has_min_time && warmup_stabilized(warmup_times)) {
        result.stabilized = true;
        break;
      }
    }

    for (int iter = 0; iter < benchmark_iters(); ++iter) {
      measure_once(true);
    }

    return result;
  }

  class GpuStageProfiler
  {
   public:
    enum class Stage : std::size_t {
      TotalBuild,
      Leaves,
      MortonCodes,
      Sort,
      Build,
      RayTracing,
      Count,
      Conversion,
      FillIndices,
      Hierarchy,
      SceneAABB,
      InternalNodesAABB,
      PrimitivesAABB,
      Total,
      _NumStages,
    };

    struct EventPair {
      cudaEvent_t start = nullptr;
      cudaEvent_t stop = nullptr;

      EventPair()
      {
        CUDA_SAFE_CALL(cudaEventCreate(&start));
        CUDA_SAFE_CALL(cudaEventCreate(&stop));
      }

      ~EventPair()
      {
        if (start) CUDA_SAFE_CALL(cudaEventDestroy(start));
        if (stop) CUDA_SAFE_CALL(cudaEventDestroy(stop));
      }

      EventPair(const EventPair&) = delete;
      EventPair& operator=(const EventPair&) = delete;

      EventPair(EventPair&& other) noexcept : start(other.start), stop(other.stop)
      {
        other.start = nullptr;
        other.stop = nullptr;
      }

      EventPair& operator=(EventPair&& other) noexcept
      {
        if (this != &other) {
          if (start) CUDA_SAFE_CALL(cudaEventDestroy(start));
          if (stop) CUDA_SAFE_CALL(cudaEventDestroy(stop));

          start = other.start;
          stop = other.stop;
          other.start = nullptr;
          other.stop = nullptr;
        }
        return *this;
      }

      float elapsed_ms() const
      {
        float ms = 0.0f;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, start, stop));
        return ms;
      }
    };

    struct StageData {
      EventPair events;
      std::vector<double> samples;
    };

    explicit GpuStageProfiler(cudaStream_t stream, std::size_t reserve_samples = 0) : stream_(stream)
    {
      for (auto& s : stages_) {
        s.samples.reserve(reserve_samples);
      }
    }

    GpuStageProfiler(const GpuStageProfiler&) = delete;
    GpuStageProfiler& operator=(const GpuStageProfiler&) = delete;
    GpuStageProfiler(GpuStageProfiler&&) = delete;
    GpuStageProfiler& operator=(GpuStageProfiler&&) = delete;

    void record_start(Stage stage) { CUDA_SAFE_CALL(cudaEventRecord(data(stage).events.start, stream_)); }

    void record_stop(Stage stage) { CUDA_SAFE_CALL(cudaEventRecord(data(stage).events.stop, stream_)); }

    double elapsed_ms(Stage stage) const { return data(stage).events.elapsed_ms(); }

    void collect(Stage stage) { data(stage).samples.push_back(elapsed_ms(stage)); }

    void collect(std::initializer_list<Stage> stages)
    {
      for (Stage s : stages) {
        collect(s);
      }
    }

    const std::vector<double>& samples(Stage stage) const { return data(stage).samples; }

    double median(Stage stage) const { return stats::median(data(stage).samples); }

    void cuda_sync_event(Stage stage) const { CUDA_SAFE_CALL(cudaEventSynchronize(data(stage).events.stop)); }

    cudaStream_t stream() const { return stream_; }

    static constexpr std::string_view name(Stage stage)
    {
      switch (stage) {
        case Stage::Total:
          return "total";
        case Stage::TotalBuild:
          return "total build";
        case Stage::SceneAABB:
          return "scene aabb";
        case Stage::Leaves:
          return "leaves";
        case Stage::MortonCodes:
          return "morton";
        case Stage::Sort:
          return "sort";
        case Stage::Build:
          return "build";
        case Stage::RayTracing:
          return "rt";
        case Stage::Count:
          return "count";
        case Stage::Conversion:
          return "conversion";
        case Stage::FillIndices:
          return "fill indices";
        case Stage::Hierarchy:
          return "hierarchy";
        case Stage::InternalNodesAABB:
          return "internal nodes aabb";
        default:
          return "unknown";
      }
    }

    void print(Stage stage, const char* prefix) const
    {
      std::cout << prefix << " " << name(stage) << " times (in ms) - " << median(stage) << std::endl;
    }

   private:
    static constexpr std::size_t index(Stage s) { return static_cast<std::size_t>(s); }

    StageData& data(Stage s) { return stages_[index(s)]; }

    const StageData& data(Stage s) const { return stages_[index(s)]; }

    cudaStream_t stream_ = nullptr;
    std::array<StageData, static_cast<std::size_t>(Stage::_NumStages)> stages_;
  };
}  // namespace benchmark
