#include "camera_reader.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "libbase/runtime_assert.h"

// -------------------- tiny XML helpers --------------------

static std::string find_tag_open_to_gt(const std::string& xml, const std::string& tagName)
{
  const std::string open = "<" + tagName;
  size_t p = xml.find(open);
  rassert(p != std::string::npos, 5101, tagName);
  size_t q = xml.find('>', p);
  rassert(q != std::string::npos, 5102, tagName);
  return xml.substr(p, q - p + 1);
}

static std::string get_attr(const std::string& tag, const std::string& name)
{
  const std::string key = name + "=\"";
  size_t p = tag.find(key);
  rassert(p != std::string::npos, 5201, name);
  p += key.size();
  size_t q = tag.find('"', p);
  rassert(q != std::string::npos && q > p, 5202, name);
  return tag.substr(p, q - p);
}

template <class T>
static std::vector<T> parse_list(const std::string& s)
{
  std::istringstream ss(s);
  std::vector<T> out;
  T v{};
  while (ss >> v) out.push_back(v);
  return out;
}

// -------------------- core --------------------

CameraView parse_view_state_from_string(const std::string& xml)
{
  const std::string cam_tag = find_tag_open_to_gt(xml, "VCGCamera");
  const std::string view_tag = find_tag_open_to_gt(xml, "ViewSettings");

  CameraView out = {};
  out.magic_bits_guard = CAMERA_VIEW_MAGIC_BITS_GUARD;

  // ---- Intrinsics ----
  {
    const float focal_mm = std::stof(get_attr(cam_tag, "FocalMm"));
    const auto pix_mm = parse_list<float>(get_attr(cam_tag, "PixelSizeMm"));      // sx sy  [mm/px]
    const auto center = parse_list<float>(get_attr(cam_tag, "CenterPx"));         // cx cy  [px]
    const auto viewport = parse_list<uint32_t>(get_attr(cam_tag, "ViewportPx"));  // w h    [px]

    rassert(pix_mm.size() == 2, 5401);
    rassert(center.size() == 2, 5402);
    rassert(viewport.size() == 2, 5403);
    rassert(pix_mm[0] != 0.0f && pix_mm[1] != 0.0f, 5404);

    out.K.focal_mm = focal_mm;
    out.K.pixel_size_mm[0] = pix_mm[0];
    out.K.pixel_size_mm[1] = pix_mm[1];

    out.K.fx = focal_mm / pix_mm[0];  // [px]
    out.K.fy = focal_mm / pix_mm[1];  // [px]
    out.K.cx = center[0];
    out.K.cy = center[1];
    out.K.width = viewport[0];
    out.K.height = viewport[1];
  }

  // --- Extrinsics ---
  const auto Rlist = parse_list<float>(get_attr(cam_tag, "RotationMatrix"));
  rassert(Rlist.size() == 9 || Rlist.size() == 16, 5501);
  float R_wc[9];
  if (Rlist.size() == 16) {
    R_wc[0] = Rlist[0];
    R_wc[1] = Rlist[4];
    R_wc[2] = Rlist[8];
    R_wc[3] = Rlist[1];
    R_wc[4] = Rlist[5];
    R_wc[5] = Rlist[9];
    R_wc[6] = Rlist[2];
    R_wc[7] = Rlist[6];
    R_wc[8] = Rlist[10];
  } else {
    for (int i = 0; i < 9; ++i) R_wc[i] = Rlist[(size_t)i];
  }

  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) out.E.R[row * 3 + col] = R_wc[col * 3 + row];

  const auto Tlist = parse_list<float>(get_attr(cam_tag, "TranslationVector"));
  rassert(Tlist.size() >= 3, 5502);
  out.E.C[0] = -Tlist[0];
  out.E.C[1] = -Tlist[1];
  out.E.C[2] = -Tlist[2];

  // t = -R * C
  out.E.t[0] = -(out.E.R[0] * out.E.C[0] + out.E.R[1] * out.E.C[1] + out.E.R[2] * out.E.C[2]);
  out.E.t[1] = -(out.E.R[3] * out.E.C[0] + out.E.R[4] * out.E.C[1] + out.E.R[5] * out.E.C[2]);
  out.E.t[2] = -(out.E.R[6] * out.E.C[0] + out.E.R[7] * out.E.C[1] + out.E.R[8] * out.E.C[2]);

  // ---- View settings ----
  {
    out.view.near_plane = std::stof(get_attr(view_tag, "NearPlane"));
    out.view.far_plane = std::stof(get_attr(view_tag, "FarPlane"));
    out.view.track_scale = std::stof(get_attr(view_tag, "TrackScale"));
  }

  return out;
}

CameraView load_view_state(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  rassert(in.good(), 5000, path);
  std::string xml((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return parse_view_state_from_string(xml);
}

std::string dump_view_state_to_string(const CameraView& camera)
{
  float Rwc[9];
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) Rwc[row * 3 + col] = camera.E.R[col * 3 + row];

  float Cx = camera.E.C[0], Cy = camera.E.C[1], Cz = camera.E.C[2];
  const bool c_all_zero = (Cx == 0.0f && Cy == 0.0f && Cz == 0.0f && (camera.E.t[0] != 0.0f || camera.E.t[1] != 0.0f || camera.E.t[2] != 0.0f));
  if (c_all_zero) {
    const float* R = camera.E.R;
    Cx = -(R[0] * camera.E.t[0] + R[3] * camera.E.t[1] + R[6] * camera.E.t[2]);
    Cy = -(R[1] * camera.E.t[0] + R[4] * camera.E.t[1] + R[7] * camera.E.t[2]);
    Cz = -(R[2] * camera.E.t[0] + R[5] * camera.E.t[1] + R[8] * camera.E.t[2]);
  }

  float M[16] = {Rwc[0], Rwc[3], Rwc[6], 0.0f, Rwc[1], Rwc[4], Rwc[7], 0.0f, Rwc[2], Rwc[5], Rwc[8], 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

  std::ostringstream os;
  os.imbue(std::locale::classic());

  os << "<!DOCTYPE ViewState>\n";
  os << "<project>\n";
  os << " <VCGCamera";
  os << " LensDistortion=\"0 0\"";
  os << " BinaryData=\"0\"";
  os << " FocalMm=\"" << camera.K.focal_mm << "\"";
  os << " RotationMatrix=\"";
  for (int i = 0; i < 16; ++i) {
    os << M[i];
    if (i + 1 < 16)
      os << ' ';
    else
      os << '"';
  }
  os << " CenterPx=\"" << camera.K.cx << ' ' << camera.K.cy << "\"";
  os << " TranslationVector=\"" << -Cx << ' ' << -Cy << ' ' << -Cz << " 1\"";
  os << " ViewportPx=\"" << camera.K.width << ' ' << camera.K.height << "\"";
  os << " CameraType=\"0\"";
  os << " PixelSizeMm=\"" << camera.K.pixel_size_mm[0] << ' ' << camera.K.pixel_size_mm[1] << "\"";

  os << "/>\n";

  // View settings
  os << " <ViewSettings"
     << " FarPlane=\"" << camera.view.far_plane << "\""
     << " NearPlane=\"" << camera.view.near_plane << "\""
     << " TrackScale=\"" << camera.view.track_scale << "\"/>\n";

  os << "</project>\n";

  return os.str();
}
