#pragma once

#include <libbase/stats.h>
#include <libbase/timer.h>
#include "libimages/images.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 10

struct RayTracingResult {
  image32i face_ids;
  image32f ao;
};

struct ConfidenceInterval {
  double lower = 0.0;
  double upper = 0.0;
};

namespace experiment_stats {

inline double percentile_from_sorted(const std::vector<double>& sorted_values, const double percentile)
{
  if (sorted_values.empty())
    return 0.0;

  const double clamped_percentile = std::clamp(percentile, 0.0, 1.0);
  const size_t index = static_cast<size_t>(clamped_percentile * static_cast<double>(sorted_values.size() - 1));
  return sorted_values[index];
}

template <typename Estimator>
inline ConfidenceInterval bootstrap_ci(const std::vector<double>& samples, Estimator estimator, const size_t bootstrap_iters = 2000)
{
  if (samples.empty())
    return {};

  if (samples.size() == 1) {
    const double value = estimator(samples);
    return {value, value};
  }

  std::mt19937 rng(123456789u);
  std::uniform_int_distribution<size_t> sample_index(0, samples.size() - 1);
  std::vector<double> resample(samples.size());
  std::vector<double> estimates;
  estimates.reserve(bootstrap_iters);

  for (size_t iter = 0; iter < bootstrap_iters; ++iter) {
    for (double& value : resample) {
      value = samples[sample_index(rng)];
    }
    estimates.push_back(estimator(resample));
  }

  std::sort(estimates.begin(), estimates.end());
  return {
      percentile_from_sorted(estimates, 0.025),
      percentile_from_sorted(estimates, 0.975),
  };
}

template <typename Estimator>
inline ConfidenceInterval bootstrap_ci(
    const std::vector<double>& lhs_samples,
    const std::vector<double>& rhs_samples,
    Estimator estimator,
    const size_t bootstrap_iters = 2000)
{
  if (lhs_samples.empty() || rhs_samples.empty())
    return {};

  if (lhs_samples.size() == 1 && rhs_samples.size() == 1) {
    const double value = estimator(lhs_samples, rhs_samples);
    return {value, value};
  }

  std::mt19937 rng(123456789u);
  std::uniform_int_distribution<size_t> lhs_index(0, lhs_samples.size() - 1);
  std::uniform_int_distribution<size_t> rhs_index(0, rhs_samples.size() - 1);
  std::vector<double> lhs_resample(lhs_samples.size());
  std::vector<double> rhs_resample(rhs_samples.size());
  std::vector<double> estimates;
  estimates.reserve(bootstrap_iters);

  for (size_t iter = 0; iter < bootstrap_iters; ++iter) {
    for (double& value : lhs_resample) {
      value = lhs_samples[lhs_index(rng)];
    }
    for (double& value : rhs_resample) {
      value = rhs_samples[rhs_index(rng)];
    }
    estimates.push_back(estimator(lhs_resample, rhs_resample));
  }

  std::sort(estimates.begin(), estimates.end());
  return {
      percentile_from_sorted(estimates, 0.025),
      percentile_from_sorted(estimates, 0.975),
  };
}

inline std::string format_ci(const ConfidenceInterval& ci, const std::string& units)
{
  std::ostringstream ss;
  ss << "95% CI [" << ci.lower << ", " << ci.upper << "]";
  if (!units.empty())
    ss << ' ' << units;
  return ss.str();
}

template <typename Fn>
inline std::vector<double> benchmark_samples(Fn&& fn, const int warmup_iters = WARMUP_ITERS, const int benchmark_iters = BENCHMARK_ITERS)
{
  std::vector<double> samples;
  samples.reserve(benchmark_iters);
  for (int iter = 0; iter < warmup_iters + benchmark_iters; ++iter) {
    timer t;
    fn();
    if (iter >= warmup_iters) {
      samples.push_back(t.elapsed());
    }
  }
  return samples;
}

inline void print_time_stats(const std::string_view label, const std::vector<double>& samples)
{
  const auto time_ci = bootstrap_ci(samples, [](const std::vector<double>& values) { return stats::median(values); });
  std::cout << label << " times (in seconds) - " << stats::valuesStatsLine(samples) << ", " << format_ci(time_ci, "seconds") << std::endl;
}

inline void print_throughput_stats(const std::string_view label, const std::vector<double>& samples, const double work_items, const std::string_view units)
{
  const double throughput = work_items * 1e-6 / stats::median(samples);
  const auto throughput_ci = bootstrap_ci(samples, [work_items](const std::vector<double>& values) { return work_items * 1e-6 / stats::median(values); });
  std::cout << label << " performance: " << throughput << ' ' << units << ", " << format_ci(throughput_ci, std::string(units)) << std::endl;
}

inline void print_phase_stats(const std::string_view label, const std::vector<double>& samples, const double work_items, const std::string_view units)
{
  print_time_stats(label, samples);
  print_throughput_stats(label, samples, work_items, units);
}

inline void print_total_time_stats(
    const std::string_view label,
    const std::vector<double>& lhs_samples,
    const std::vector<double>& rhs_samples)
{
  const double total_time = stats::median(lhs_samples) + stats::median(rhs_samples);
  const auto total_time_ci = bootstrap_ci(lhs_samples, rhs_samples, [](const std::vector<double>& lhs, const std::vector<double>& rhs) {
    return stats::median(lhs) + stats::median(rhs);
  });
  std::cout << label << ": " << total_time << " seconds, " << format_ci(total_time_ci, "seconds") << std::endl;
}

inline std::vector<double> single_sample(const double value) { return {value}; }

}  // namespace experiment_stats
