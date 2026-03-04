#pragma once

#include "../experiments/common.h"
#include "../kernels/defines.h"
#include "../kernels/structs/bvh_node_gpu.h"
#include "cuda_utils.h"
#include "libimages/debug_io.h"
#include "libimages/images.h"

inline int divCeil(int a, int b) { return (a + b - 1) / b; }

inline size_t divCeil(size_t a, size_t b)
{
  rassert(a + b > 0, 131997010);
  return (a + b - 1) / b;
}

inline size_t compute_grid(size_t n) { return std::min(divCeil(n, (size_t)DEFAULT_GROUP_SIZE), (size_t)MAX_GRID_SIZE); }

inline dim3 compute_grid(size_t x, size_t y)
{
  return dim3(
      std::min(divCeil(x, (size_t)DEFAULT_GROUP_SIZE_X), (size_t)MAX_GRID_SIZE_X),
      std::min(divCeil(y, (size_t)DEFAULT_GROUP_SIZE_Y), (size_t)MAX_GRID_SIZE_Y));
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
  rassert(face_errors < width * height / 99, 3453415123546587ULL, face_errors, to_percent(face_errors, width * height));
}

// Surface Area Heuristic - метрика качества BVH
// Меньше — лучше: чем ниже значение, тем меньше ожидаемое число операций при трассировке случайного луча.
inline void report_sah_h(const std::vector<BVHNodeGPU>& lbvh_nodes)
{
  const int n_total = lbvh_nodes.size();
  const int n_faces = (n_total + 1) / 2;

  // Площадь поверхности AABB
  auto surfaceArea = [](const AABBGPU& b) -> double {
    double dx = static_cast<double>(b.max_x) - b.min_x;
    double dy = static_cast<double>(b.max_y) - b.min_y;
    double dz = static_cast<double>(b.max_z) - b.min_z;
    return 2.0 * (dx * dy + dy * dz + dz * dx);
  };

  // Типовые константы из литературы (Wald et al., Aila & Laine)
  const double C_trav = 1.2;   // стоимость обхода внутреннего узла
  const double C_isect = 1.0;  // стоимость пересечения луч-треугольник

  // Разметка узлов в стандартном LBVH (n = nfaces листьев):
  //   [0, nfaces-2]  — внутренние узлы (nfaces-1 штук)
  //   [nfaces-1, 2*nfaces-2]— листья           (nfaces штук, по 1 треуг.)
  //   Корень = узел 0
  const unsigned int n_internal = n_faces - 1;

  const double root_sa = surfaceArea(lbvh_nodes[0].aabb);

  if (root_sa <= 0.0) {
    std::cerr << "SAH is undefined: root surface area <= 0\n";
  } else {
    double sum_internal_sa = 0.0;
    double sum_leaf_sa = 0.0;

    for (unsigned int i = 0; i < n_total; ++i) {
      double sa = surfaceArea(lbvh_nodes[i].aabb);
      if (i < n_internal)
        sum_internal_sa += sa;
      else
        sum_leaf_sa += sa;
    }

    // Взвешенная стоимость, нормированная на SA корня
    double sah_cost = (C_trav * sum_internal_sa + C_isect * sum_leaf_sa) / root_sa;

    // Дополнительные компоненты для диагностики
    double norm_internal = sum_internal_sa / root_sa;
    double norm_leaves = sum_leaf_sa / root_sa;

    std::cout << "BVH SAH quality metrics:\n";
    std::cout << "  C_trav=" << C_trav << "  C_isect=" << C_isect << "\n";
    std::cout << "  Internal nodes : " << n_internal << "  leaves: " << n_faces << "\n";
    std::cout << "  SA(internal) / SA(root) = " << norm_internal << "  (cost in SAH: " << C_trav * norm_internal << ")\n";
    std::cout << "  SA(leaves)   / SA(root) = " << norm_leaves << "  (cost in SAH: " << C_isect * norm_leaves << ")\n";
    std::cout << "  SAH cost = " << sah_cost << std::endl;
  }
}

// Surface Area Heuristic - метрика качества BVH
// Меньше — лучше: чем ниже значение, тем меньше ожидаемое число операций при трассировке случайного луча.
inline void report_sah_d(cudaStream_t stream, unsigned int nfaces, BVHNodeGPU* d_lbvh_nodes)
{
  // Читаем все узлы с GPU на CPU (однократно, вне петли замеров)
  const unsigned int n_total = 2 * nfaces - 1;
  std::vector<BVHNodeGPU> h_nodes(n_total);
  CUDA_SAFE_CALL(cudaMemcpyAsync(h_nodes.data(), d_lbvh_nodes, sizeof(BVHNodeGPU) * n_total, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK_STREAM(stream);
  report_sah_h(h_nodes);
}
