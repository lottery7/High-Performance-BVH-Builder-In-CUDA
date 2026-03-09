#pragma once

#include <cuda_runtime_api.h>

#include <numeric>
#include <string>

#include "../experiments/common.h"
#include "../kernels/structs/bvh_node.h"
#include "defines.h"
#include "exceptions.h"
#include "libbase/string_utils.h"
#include "libimages/debug_io.h"
#include "libimages/images.h"

#define CUDA_SAFE_CALL(expr) cuda::report_error(expr, __LINE__)
#define CUDA_SYNC_STREAM(expr) cuda::report_error(cudaStreamSynchronize(expr), __LINE__)

#if RASSERT_ENABLED
#define curassert(condition, error_code)                                      \
  do {                                                                        \
    if (!(condition)) {                                                       \
      printf("rassert code=%d line=%d\n", error_code % 1000000000, __LINE__); \
    }                                                                         \
  } while (false)
#else
#define curassert(condition, error_code)  // do nothing
#endif

namespace cuda
{
  inline ::std::string format_error(cudaError_t code) { return ::std::string(cudaGetErrorString(code)) + " (" + ::std::to_string(code) + ")"; }

  inline void report_error(cudaError_t err, int line, const ::std::string& prefix = ::std::string())
  {
    if (cudaSuccess == err) return;

    auto message = prefix + format_error(err) + " at line " + ::std::to_string(line);

    size_t total_mem_size = 0;
    size_t free_mem_size = 0;
    cudaError_t err2;

    switch (err) {
      case cudaErrorMemoryAllocation:
        err2 = cudaMemGetInfo(&free_mem_size, &total_mem_size);
        if (cudaSuccess == err2)
          message = message + "(free memory: " + ::std::to_string(free_mem_size >> 20) + "/" + ::std::to_string(total_mem_size >> 20) + " MB)";
        else
          message = message + "(free memory unknown: " + format_error(err2) + ")";
        throw CudaBadAlloc(message);
      default:
        throw CudaException(message);
    }
  }

  inline void select_cuda_device(int argc, char** argv)
  {
    int device_count = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) throw ::std::runtime_error("No CUDA devices available");
    int device_id = 0;
    if (device_count > 1) {
      if (argc < 2) throw ::std::runtime_error("Multiple GPUs available. Pass device index as argument.");
      device_id = ::std::stoi(argv[1]);
      if (device_id < 0 || device_id >= device_count) throw ::std::runtime_error("Invalid device index");
    }
    CUDA_SAFE_CALL(cudaSetDevice(device_id));
  }
}  // namespace cuda

inline int div_ceil(int a, int b) { return (a + b - 1) / b; }

inline size_t div_ceil(size_t a, size_t b)
{
  rassert(a + b > 0, 131997010);
  return (a + b - 1) / b;
}

inline size_t compute_grid(size_t n) { return std::min(div_ceil(n, static_cast<size_t>(DEFAULT_GROUP_SIZE)), static_cast<size_t>(MAX_GRID_SIZE)); }

inline dim3 compute_grid(size_t x, size_t y)
{
  return dim3(
      std::min(div_ceil(x, static_cast<size_t>(DEFAULT_GROUP_SIZE_X)), static_cast<size_t>(MAX_GRID_SIZE_X)),
      std::min(div_ceil(y, static_cast<size_t>(DEFAULT_GROUP_SIZE_Y)), static_cast<size_t>(MAX_GRID_SIZE_Y)));
}

template <typename T>
size_t count_non_empty(const TypedImage<T>& image, T empty_value)
{
  rassert(image.channels() == 1, 4523445132412, image.channels());
  size_t count = 0;
#pragma omp parallel for reduction(+ : count)
  for (ptrdiff_t j = 0; j < image.height(); ++j)
    for (ptrdiff_t i = 0; i < image.width(); ++i)
      if (image.ptr(j)[i] != empty_value) ++count;
  return count;
}

template <typename T>
size_t count_diffs(const TypedImage<T>& a, const TypedImage<T>& b, T threshold)
{
  rassert(a.channels() == 1, 5634532413241, a.channels());
  rassert(a.channels() == b.channels(), 562435231453243);
  rassert(a.width() == b.width() && a.height() == b.height(), 562435231453243);
  size_t count = 0;
#pragma omp parallel for reduction(+ : count)
  for (ptrdiff_t j = 0; j < a.height(); ++j)
    for (ptrdiff_t i = 0; i < a.width(); ++i)
      if (std::abs(a.ptr(j)[i] - b.ptr(j)[i]) > threshold) ++count;
  return count;
}

inline void save_framebuffers(const std::string& results_dir, const std::string& suffix, const image32i& face_ids, const image32f& ao)
{
  debug_io::dumpImage(results_dir + "/framebuffer_face_ids_" + suffix + ".bmp", debug_io::randomMapping(face_ids, NO_FACE_ID));
  debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_" + suffix + ".bmp", debug_io::depthMapping(ao));
}

inline void validate_against_ground_truth(
    const RayTracingResult& ground_truth_res,
    const RayTracingResult& cmp_res,
    unsigned int width,
    unsigned int height)
{
  unsigned int ao_errors = count_diffs(ground_truth_res.ao, cmp_res.ao, 0.01f);
  unsigned int face_errors = count_diffs(ground_truth_res.face_ids, cmp_res.face_ids, 1);
  rassert(ao_errors < width * height / 100, 345341512354123ULL, ao_errors, to_percent(ao_errors, width * height));
  rassert(face_errors < width * height / 100, 3453415123546587ULL, face_errors, to_percent(face_errors, width * height));
}

inline void report_sah(const std::vector<BVHNode>& bvh_nodes)
{
  const int n_total = bvh_nodes.size();
  const int n_faces = (n_total + 1) / 2;
  const unsigned int leaves_start = n_faces - 1;

  for (int i = 0; i < n_total; ++i) {
    auto& n = bvh_nodes[i];
    if (i < leaves_start) {  // как вы считаете internal
      if (n.left_child_index >= n_total || n.right_child_index >= n_total) {
        std::printf("Bad child at node %d: L=%d R=%d\n", i, n.left_child_index, n.right_child_index);
        break;
      }
    }
  }

  constexpr float C_trav = 2;   // NOLINT(*-identifier-naming)
  constexpr float C_isect = 3;  // NOLINT(*-identifier-naming)

  float sah = 0;
  for (int i = 0; i < leaves_start; i++) sah += C_trav * bvh_nodes[i].aabb.surface_area();
  for (int i = leaves_start; i < n_total; i++) sah += C_isect * bvh_nodes[i].aabb.surface_area();
  sah /= bvh_nodes[0].aabb.surface_area();

  std::cout << "SAH = " << sah << " (C_trav=" << C_trav << ", C_isect=" << C_isect << ")" << std::endl;
}

inline void report_sah(cudaStream_t stream, const BVHNode* d_bvh, unsigned int n_nodes)
{
  std::vector<BVHNode> h_nodes(n_nodes);
  CUDA_SAFE_CALL(cudaMemcpyAsync(h_nodes.data(), d_bvh, sizeof(BVHNode) * n_nodes, cudaMemcpyDeviceToHost, stream));
  CUDA_SYNC_STREAM(stream);
  report_sah(h_nodes);
}
