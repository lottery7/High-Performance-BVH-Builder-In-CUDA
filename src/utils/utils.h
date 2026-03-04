#pragma once

#include "../experiments/common.h"
#include "../kernels/defines.h"
#include "libimages/debug_io.h"
#include "libimages/images.h"

struct RayTracingResult;
inline int divCeil(int a, int b) { return (a + b - 1) / b; }
inline size_t divCeil(size_t a, size_t b)
{
  rassert(a + b > 0, 131997010);
  return (a + b - 1) / b;
}

template <typename T>
size_t countNonEmpty(const TypedImage<T>& image, T empty_value)
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
size_t countDiffs(const TypedImage<T>& a, const TypedImage<T>& b, T threshold)
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

inline void saveFramebuffers(const std::string& results_dir, const std::string& suffix, const image32i& face_ids, const image32f& ao)
{
  debug_io::dumpImage(results_dir + "/framebuffer_face_ids_" + suffix + ".bmp", debug_io::randomMapping(face_ids, NO_FACE_ID));
  debug_io::dumpImage(results_dir + "/framebuffer_ambient_occlusion_" + suffix + ".bmp", debug_io::depthMapping(ao));
}

inline void validateAgainstGroundTruth(
    const RayTracingResult& ground_truth_res,
    const RayTracingResult& cmp_res,
    unsigned int width,
    unsigned int height)
{
  unsigned int ao_errors = countDiffs(ground_truth_res.ao, cmp_res.ao, 0.01f);
  unsigned int face_errors = countDiffs(ground_truth_res.face_ids, cmp_res.face_ids, 1);
  rassert(ao_errors < width * height / 100, 345341512354123ULL, ao_errors, to_percent(ao_errors, width * height));
  rassert(face_errors < width * height / 100, 3453415123546587ULL, face_errors, to_percent(face_errors, width * height));
}
