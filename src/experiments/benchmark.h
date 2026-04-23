#pragma once

#include <cuda_runtime_api.h>
#include <libbase/stats.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
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
  double measure(cudaStream_t stream, F&& fn)
  {
    CUDA_SAFE_CALL(cudaEventRecord(start_, stream));
    std::forward<F>(fn)();
    CUDA_SAFE_CALL(cudaEventRecord(stop_, stream));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop_));

    float elapsed_ms = 0.0f;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_ms, start_, stop_));
    return static_cast<double>(elapsed_ms);
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
}  // namespace benchmark
