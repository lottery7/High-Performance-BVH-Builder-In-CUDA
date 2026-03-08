#pragma once

/* Cross-kernel, padding-free camera structs.
   Rules:
   - Only 32-bit scalars: float and unsigned 32-bit integer.
   - Explicit arrays, no vec3/vec4/bool/double.
   - Row-major conventions are fixed in comments below.
   - No alignment/pragmas required; every member has 4-byte alignment.
   - Can be included from CUDA/OpenCL C/host C++.
*/

/* -------- Intrinsics (pixels) --------
   fx, fy, cx, cy — стандартные параметры.
   pixel_size_mm[2] — физический размер пикселя (mm) по X/Y.
   focal_mm — фокус в мм (удобно иметь под рукой).
   width, height — габариты изображения в пикселях.
*/
struct CameraIntrinsics {
  float fx;               /* [px] */
  float fy;               /* [px] */
  float cx;               /* [px] */
  float cy;               /* [px] */
  float pixel_size_mm[2]; /* [mm] {sx, sy} */
  float focal_mm;         /* [mm] */
  unsigned int width;     /* [px] */
  unsigned int height;    /* [px] */
};

/* -------- Extrinsics --------
   X_cam = R * X_world + t
   R — 3x3 row-major: R = [ r00 r01 r02
                            r10 r11 r12
                            r20 r21 r22 ]
   t — перевод в камеру.
   C — центр камеры в мире: C = -R^T * t (заполните на хосте, если нужно).
*/
struct CameraExtrinsics {
  float R[9]; /* row-major 3x3: r00,r01,r02, r10,r11,r12, r20,r21,r22 */
  float t[3]; /* translation */
  float C[3]; /* camera center (world), optional to use */
};

/* -------- View (прочие настройки проекции/визуализации) -------- */
struct ViewSettings {
  float near_plane;
  float far_plane;
  float track_scale;
};

#define CAMERA_VIEW_MAGIC_BITS_GUARD 239239239

/* -------- Full packed view -------- */
struct CameraView {
  CameraIntrinsics K;
  CameraExtrinsics E;
  ViewSettings view;
  unsigned int magic_bits_guard;
};

/* ---------------- Host-only layout checks ---------------- */
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(unsigned int) == 4, "unsigned int must be 32-bit");

static_assert(sizeof(CameraIntrinsics) == (4 * 7 + 4 * 2), "Intrinsics size mismatch"); /* 7 floats + 2 uints = 36 bytes */
static_assert(sizeof(CameraExtrinsics) == (9 + 3 + 3) * 4, "Extrinsics size mismatch");
static_assert(sizeof(ViewSettings) == 3 * 4, "ViewSettings size mismatch");
static_assert(sizeof(CameraView) == sizeof(CameraIntrinsics) + sizeof(CameraExtrinsics) + sizeof(ViewSettings) + 4, "CameraView size mismatch");
