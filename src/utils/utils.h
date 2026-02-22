#pragma once
#include "libimages/images.h"

inline int divCeil(int a, int b) { return (a + b - 1) / b; }

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
