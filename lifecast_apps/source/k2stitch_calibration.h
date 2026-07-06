// MIT License. k2stitch: rig calibration model for Z CAM K2 Pro (or any dual-fisheye
// two-file VR180 rig) -> Lifecast VVE ingest format.
//
// Angle fields are stored in DEGREES (human-editable JSON); they are converted to
// radians only when building rotation matrices. Distances are in meters.
//
// Coordinate convention (matches the rest of lifecast_apps): +X right, +Y up, +Z forward.
// world_from_cam = Ry(yaw) * Rx(pitch) * Rz(roll).
//
// The math here is validated end-to-end against Lifecast's own projection code by
// k2stitch_test_math.py (T1/T2/T3).

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "fisheye_camera.h"
#include "logger.h"
#include "third_party/json.h"

namespace p11 { namespace k2stitch {

constexpr double kDegToRad = M_PI / 180.0;

struct EyeCalibration {
  // Raw fisheye intrinsics, in source-video pixel coordinates.
  double center_x = 0.0;        // optical center (0 => auto: width/2)
  double center_y = 0.0;        // optical center (0 => auto: height/2)
  double circle_radius = 0.0;   // image circle radius for the FULL lens FOV (0 => auto)
  double lens_fov_deg = 200.0;  // iZugar MKX200 = 200 degrees
  double k1 = 0.0;              // radial distortion (FisheyeCamera model)
  double tilt = 0.0;            // sensor/lens tilt ovalness (FisheyeCamera model)
  // Extrinsics: rotation of this camera relative to an ideal forward-looking rig [deg].
  double yaw = 0.0;
  double pitch = 0.0;
  double roll = 0.0;
};

struct RigCalibration {
  EyeCalibration left, right;
  double global_roll = 0.0;      // horizon correction applied to both eyes [deg]
  double convergence = 0.0;      // symmetric toe-in (+) / toe-out (-) [deg]
  int sync_offset_frames = 0;    // right video leads left by this many frames
  bool skip_first_frame = true;  // K2 Pro slave-unit first-frame exposure quirk
  bool swap_eyes = false;        // insurance against mislabeled L/R media
  double ipd_m = 0.064;          // K2 Pro fixed baseline; use 6.4 cm in VVE
  double gain_bgr[3] = {1.0, 1.0, 1.0};  // color match gains applied to the LEFT (slave) eye
};

inline Eigen::Matrix3d rotX(const double a)
{
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitX()).toRotationMatrix();
}
inline Eigen::Matrix3d rotY(const double a)
{
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitY()).toRotationMatrix();
}
inline Eigen::Matrix3d rotZ(const double a)
{
  return Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

inline Eigen::Matrix3d worldFromCamYPR(
    const double yaw_deg, const double pitch_deg, const double roll_deg)
{
  return rotY(yaw_deg * kDegToRad) * rotX(pitch_deg * kDegToRad) * rotZ(roll_deg * kDegToRad);
}

// Build a lifecast FisheyeCamerad for one eye. extra_yaw_deg carries convergence
// (+conv/2 for the left eye, -conv/2 for the right eye); global_roll applies to both.
inline calibration::FisheyeCamerad makeEyeCamera(
    const EyeCalibration& eye,
    const int video_width,
    const int video_height,
    const double extra_yaw_deg,
    const double global_roll_deg)
{
  calibration::FisheyeCamerad cam;
  cam.width = video_width;
  cam.height = video_height;
  const double cx = eye.center_x > 0 ? eye.center_x : video_width / 2.0;
  const double cy = eye.center_y > 0 ? eye.center_y : video_height / 2.0;
  const double circle_r =
      eye.circle_radius > 0 ? eye.circle_radius : std::min(video_width, video_height) * 0.4875;
  cam.optical_center = Eigen::Vector2d(cx, cy);
  // f-theta: r = phi * radius_at_90 / (pi/2)  =>  radius_at_90 = circle_r * 90 / (fov/2)
  cam.radius_at_90 = circle_r * 90.0 / (eye.lens_fov_deg / 2.0);
  cam.k1 = eye.k1;
  cam.tilt = eye.tilt;
  cam.useable_radius = circle_r;
  const Eigen::Matrix3d world_from_cam = worldFromCamYPR(
      eye.yaw + extra_yaw_deg, eye.pitch, eye.roll + global_roll_deg);
  cam.cam_from_world.linear() = world_from_cam.transpose();
  cam.cam_from_world.translation() = Eigen::Vector3d(0, 0, 0);
  return cam;
}

inline nlohmann::json eyeToJson(const EyeCalibration& e)
{
  nlohmann::json j;
  j["center_x"] = e.center_x;
  j["center_y"] = e.center_y;
  j["circle_radius"] = e.circle_radius;
  j["lens_fov_deg"] = e.lens_fov_deg;
  j["k1"] = e.k1;
  j["tilt"] = e.tilt;
  j["yaw_deg"] = e.yaw;
  j["pitch_deg"] = e.pitch;
  j["roll_deg"] = e.roll;
  return j;
}

inline EyeCalibration eyeFromJson(const nlohmann::json& j)
{
  EyeCalibration e;
  e.center_x = j.value("center_x", 0.0);
  e.center_y = j.value("center_y", 0.0);
  e.circle_radius = j.value("circle_radius", 0.0);
  e.lens_fov_deg = j.value("lens_fov_deg", 200.0);
  e.k1 = j.value("k1", 0.0);
  e.tilt = j.value("tilt", 0.0);
  e.yaw = j.value("yaw_deg", 0.0);
  e.pitch = j.value("pitch_deg", 0.0);
  e.roll = j.value("roll_deg", 0.0);
  return e;
}

inline bool saveCalibration(const RigCalibration& c, const std::string& path)
{
  nlohmann::json j;
  j["format"] = "k2stitch_rig_calibration_v1";
  j["left"] = eyeToJson(c.left);
  j["right"] = eyeToJson(c.right);
  j["global_roll_deg"] = c.global_roll;
  j["convergence_deg"] = c.convergence;
  j["sync_offset_frames"] = c.sync_offset_frames;
  j["skip_first_frame"] = c.skip_first_frame;
  j["swap_eyes"] = c.swap_eyes;
  j["ipd_m"] = c.ipd_m;
  j["gain_bgr"] = {c.gain_bgr[0], c.gain_bgr[1], c.gain_bgr[2]};
  std::ofstream f(path);
  if (!f.is_open()) return false;
  f << j.dump(2) << std::endl;
  return true;
}

inline bool loadCalibration(const std::string& path, RigCalibration& c)
{
  std::ifstream f(path);
  if (!f.is_open()) return false;
  nlohmann::json j;
  try {
    f >> j;
    c.left = eyeFromJson(j.at("left"));
    c.right = eyeFromJson(j.at("right"));
    c.global_roll = j.value("global_roll_deg", 0.0);
    c.convergence = j.value("convergence_deg", 0.0);
    c.sync_offset_frames = j.value("sync_offset_frames", 0);
    c.skip_first_frame = j.value("skip_first_frame", true);
    c.swap_eyes = j.value("swap_eyes", false);
    c.ipd_m = j.value("ipd_m", 0.064);
    if (j.count("gain_bgr") && j["gain_bgr"].size() == 3) {
      for (int i = 0; i < 3; ++i) c.gain_bgr[i] = j["gain_bgr"][i];
    }
  } catch (const std::exception& ex) {
    XPLINFO << "Failed to parse calibration JSON: " << ex.what();
    return false;
  }
  return true;
}

}}  // namespace p11::k2stitch
