#pragma once

#include "aabb.h"
#include "utils/defines.h"

struct BVH2Node {
  AABB aabb;
  unsigned int left_child_index;
  unsigned int right_child_index;

  __host__ __device__ bool is_leaf() const { return left_child_index == INVALID_INDEX; }
};

struct __align__(16) BVH8Node {
  // 1. Базовая точка локальной сетки (12 байт)
  float p_x, p_y, p_z;

  // 2. Экспоненты для масштаба (3 байта)
  uint8_t e_x, e_y, e_z;

  // 3. Маска внутренних узлов (1 байт). Бит = 1, если в слоте внутренний узел
  uint8_t imask;

  // 4. Базовые индексы (8 байт)
  unsigned int child_base_idx;  // Индекс первой дочерней ноды
  unsigned int prim_base_idx;   // Индекс первого треугольника в плотном массиве

  // 5. Метаданные слотов (8 байт)
  uint8_t meta[8];

  // 6. Сжатые границы AABB (48 байт)
  uint8_t q_lo_x[8], q_lo_y[8], q_lo_z[8];
  uint8_t q_hi_x[8], q_hi_y[8], q_hi_z[8];
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");
static_assert(sizeof(BVH2Node) == sizeof(AABB) + 2 * 4, "BVH2Node size mismatch");
