// MIT License. k2stitch core implementation. See k2stitch_core.h.
// Math validated against Lifecast's projection code by k2stitch_test_math.py.

#include "k2stitch_core.h"

#include <cmath>
#include <cstdio>

#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

#include "logger.h"
#include "util_command.h"
#include "util_file.h"
#include "util_math.h"
#include "util_string.h"

#ifdef _WIN32
#define K2_POPEN _popen
#define K2_PCLOSE _pclose
#else
#define K2_POPEN popen
#define K2_PCLOSE pclose
#endif

namespace p11 { namespace k2stitch {

namespace {

inline double verticalAngle(const Eigen::Vector3d& ray)
{
  return std::atan2(-ray.y(), std::sqrt(ray.x() * ray.x() + ray.z() * ray.z()));
}

}  // namespace

void precomputeFisheyeToVR180Warp(
    const calibration::FisheyeCamerad& cam,
    const int eqr_size,
    const double lens_fov_deg,
    std::vector<cv::Mat>& warp_uv)
{
  // Same construction as projection::precomputeremapFisheyeToEquirectWarp, with
  // explicit invalidation of rays beyond the lens coverage. Convention validated
  // against VVE's precomputeVR180toFthetaWarp in k2stitch_test_math.py (T2).
  const double max_coverage_deg = std::min(115.0, lens_fov_deg / 2.0 + 15.0);
  const double z_min = std::cos(max_coverage_deg * kDegToRad);

  cv::Mat warp(cv::Size(eqr_size, eqr_size), CV_32FC2);
  for (int y = 0; y < eqr_size; ++y) {
    for (int x = 0; x < eqr_size; ++x) {
      const double lon = -M_PI * (double(x) / eqr_size - 0.5) + M_PI / 2.0;
      const double lat = M_PI * (double(y) / eqr_size - 0.5);
      const Eigen::Vector3d p(
          std::cos(lat) * std::cos(lon), -std::sin(lat), std::cos(lat) * std::sin(lon));
      const Eigen::Vector3d p_cam = cam.camFromWorld(p);
      if (p_cam.z() < z_min) {
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(-1, -1);
        continue;
      }
      const Eigen::Vector2d pixel = cam.pixelFromCam(p_cam);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(pixel.x(), pixel.y());
    }
  }
  cv::split(warp, warp_uv);
}

void buildWarps(
    const RigCalibration& calib,
    const cv::Size& video_size,
    const int eqr_size,
    StitchWarps& warps)
{
  const calibration::FisheyeCamerad cam_L = makeEyeCamera(
      calib.left, video_size.width, video_size.height,
      /*extra_yaw_deg=*/+calib.convergence / 2.0, calib.global_roll);
  const calibration::FisheyeCamerad cam_R = makeEyeCamera(
      calib.right, video_size.width, video_size.height,
      /*extra_yaw_deg=*/-calib.convergence / 2.0, calib.global_roll);
  precomputeFisheyeToVR180Warp(cam_L, eqr_size, calib.left.lens_fov_deg, warps.warp_L);
  precomputeFisheyeToVR180Warp(cam_R, eqr_size, calib.right.lens_fov_deg, warps.warp_R);
  warps.eqr_size = eqr_size;
  warps.video_size = video_size;
}

void stitchEyes(
    const RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    cv::Mat& eq_L,
    cv::Mat& eq_R,
    const cv::InterpolationFlags interp)
{
  const cv::Mat& src_L = calib.swap_eyes ? frame_R : frame_L;
  const cv::Mat& src_R = calib.swap_eyes ? frame_L : frame_R;
  cv::remap(src_L, eq_L, warps.warp_L[0], warps.warp_L[1], interp, cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0, 0));
  cv::remap(src_R, eq_R, warps.warp_R[0], warps.warp_R[1], interp, cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0, 0));
  if (calib.gain_bgr[0] != 1.0 || calib.gain_bgr[1] != 1.0 || calib.gain_bgr[2] != 1.0) {
    cv::multiply(
        eq_L,
        cv::Scalar(calib.gain_bgr[0], calib.gain_bgr[1], calib.gain_bgr[2]),
        eq_L);
  }
}

cv::Mat stitchFrame(
    const RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    const cv::InterpolationFlags interp)
{
  cv::Mat eq_L, eq_R, sbs;
  stitchEyes(calib, warps, frame_L, frame_R, eq_L, eq_R, interp);
  cv::hconcat(eq_L, eq_R, sbs);
  return sbs;
}

cv::Mat compositeAnaglyph(const cv::Mat& eq_L, const cv::Mat& eq_R)
{
  XCHECK_EQ(eq_L.size(), eq_R.size());
  std::vector<cv::Mat> ch_L, ch_R;
  cv::split(eq_L, ch_L);
  cv::split(eq_R, ch_R);
  // BGR: blue+green from right, red from left => red=L, cyan=R
  std::vector<cv::Mat> out = {ch_R[0], ch_R[1], ch_L[2]};
  cv::Mat anaglyph;
  cv::merge(out, anaglyph);
  return anaglyph;
}

cv::Mat compositeBlend(const cv::Mat& eq_L, const cv::Mat& eq_R)
{
  cv::Mat blend;
  cv::addWeighted(eq_L, 0.5, eq_R, 0.5, 0.0, blend);
  return blend;
}

cv::Mat compositeDifference(const cv::Mat& eq_L, const cv::Mat& eq_R)
{
  cv::Mat diff;
  cv::absdiff(eq_L, eq_R, diff);
  diff *= 4.0;
  return diff;
}

void drawLatitudeGrid(cv::Mat& img, const int spacing_deg)
{
  // Lines of constant vertical angle: y = S * (0.5 + delta_deg / 180)
  const int S = img.rows;
  for (int deg = -90 + spacing_deg; deg <= 90 - spacing_deg; deg += spacing_deg) {
    const int y = int(S * (0.5 + double(deg) / 180.0));
    const cv::Scalar color = (deg == 0) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 180, 0);
    cv::line(img, cv::Point(0, y), cv::Point(img.cols - 1, y), color, 1, cv::LINE_AA);
  }
}

RefineReport refineAlignment(
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    RigCalibration& calib,
    const int solver_eqr_size)
{
  RefineReport report;

  StitchWarps warps;
  buildWarps(calib, frame_L.size(), solver_eqr_size, warps);
  cv::Mat eq_L, eq_R;
  stitchEyes(calib, warps, frame_L, frame_R, eq_L, eq_R, cv::INTER_LINEAR);

  cv::Mat gray_L, gray_R;
  cv::cvtColor(eq_L, gray_L, cv::COLOR_BGR2GRAY);
  cv::cvtColor(eq_R, gray_R, cv::COLOR_BGR2GRAY);

  // ORB feature matching (same parameters as validated in k2stitch_test_math.py T3)
  cv::Ptr<cv::ORB> orb = cv::ORB::create(3000);
  orb->setFastThreshold(12);
  std::vector<cv::KeyPoint> kp_L, kp_R;
  cv::Mat desc_L, desc_R;
  orb->detectAndCompute(gray_L, cv::noArray(), kp_L, desc_L);
  orb->detectAndCompute(gray_R, cv::noArray(), kp_R, desc_R);
  if (desc_L.empty() || desc_R.empty()) {
    report.message = "No features detected. Is the frame too dark or uniform?";
    return report;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc_L, desc_R, knn, 2);

  // The cameras used to interpret raw fisheye pixels (fixed during the solve).
  const calibration::FisheyeCamerad cam_L = makeEyeCamera(
      calib.left, frame_L.cols, frame_L.rows, +calib.convergence / 2.0, calib.global_roll);
  const calibration::FisheyeCamerad cam_R = makeEyeCamera(
      calib.right, frame_R.cols, frame_R.rows, -calib.convergence / 2.0, calib.global_roll);

  std::vector<Eigen::Vector3d> rays_L, rays_R;
  for (const auto& m2 : knn) {
    if (m2.size() < 2) continue;
    if (m2[0].distance >= 0.75f * m2[1].distance) continue;  // Lowe ratio test
    const cv::Point2f& pL = kp_L[m2[0].queryIdx].pt;
    const cv::Point2f& pR = kp_R[m2[0].trainIdx].pt;
    const int xL = math::clamp(int(std::lround(pL.x)), 0, solver_eqr_size - 1);
    const int yL = math::clamp(int(std::lround(pL.y)), 0, solver_eqr_size - 1);
    const int xR = math::clamp(int(std::lround(pR.x)), 0, solver_eqr_size - 1);
    const int yR = math::clamp(int(std::lround(pR.y)), 0, solver_eqr_size - 1);
    const float fxL = warps.warp_L[0].at<float>(yL, xL);
    const float fyL = warps.warp_L[1].at<float>(yL, xL);
    const float fxR = warps.warp_R[0].at<float>(yR, xR);
    const float fyR = warps.warp_R[1].at<float>(yR, xR);
    if (fxL < 0 || fyL < 0 || fxR < 0 || fyR < 0) continue;
    rays_L.push_back(cam_L.rayDirFromPixel(Eigen::Vector2d(fxL, fyL)));
    rays_R.push_back(cam_R.rayDirFromPixel(Eigen::Vector2d(fxR, fyR)));
  }
  report.num_matches = int(rays_L.size());
  if (report.num_matches < 40) {
    report.message = "Only " + std::to_string(report.num_matches) +
                     " feature matches; need at least 40. Try a more textured frame.";
    return report;
  }

  // Residuals as a function of (d_pitch_deg, d_roll_deg) applied to the right eye.
  const Eigen::Matrix3d wfc_L = cam_L.cam_from_world.linear().transpose();
  std::vector<double> delta_L(rays_L.size());
  for (size_t i = 0; i < rays_L.size(); ++i) {
    delta_L[i] = verticalAngle(wfc_L * rays_L[i]);
  }
  const double yaw_R = calib.right.yaw - calib.convergence / 2.0;
  const double pitch_R = calib.right.pitch;
  const double roll_R = calib.right.roll + calib.global_roll;
  auto residuals = [&](const double dp, const double dr, std::vector<double>& r) {
    const Eigen::Matrix3d wfc_R = worldFromCamYPR(yaw_R, pitch_R + dp, roll_R + dr);
    r.resize(rays_R.size());
    for (size_t i = 0; i < rays_R.size(); ++i) {
      r[i] = verticalAngle(wfc_R * rays_R[i]) - delta_L[i];
    }
  };

  // Robust pre-filter: drop outliers by median absolute deviation.
  std::vector<double> r0;
  residuals(0, 0, r0);
  const double med = math::median(r0);
  std::vector<double> abs_dev(r0.size());
  for (size_t i = 0; i < r0.size(); ++i) abs_dev[i] = std::abs(r0[i] - med);
  const double mad = math::median(abs_dev) + 1e-9;
  std::vector<Eigen::Vector3d> rays_L2, rays_R2;
  for (size_t i = 0; i < r0.size(); ++i) {
    if (std::abs(r0[i] - med) < 4.0 * mad) {
      rays_L2.push_back(rays_L[i]);
      rays_R2.push_back(rays_R[i]);
    }
  }
  rays_L.swap(rays_L2);
  rays_R.swap(rays_R2);
  delta_L.resize(rays_L.size());
  for (size_t i = 0; i < rays_L.size(); ++i) delta_L[i] = verticalAngle(wfc_L * rays_L[i]);

  residuals(0, 0, r0);
  double rms0 = 0;
  for (const double r : r0) rms0 += r * r;
  rms0 = std::sqrt(rms0 / r0.size());
  report.rms_before_px = rms0 * (180.0 / M_PI) * solver_eqr_size / 180.0;

  // Levenberg-Marquardt with IRLS Huber weights (2 parameters, numeric Jacobian).
  static constexpr double kHuber = 0.003;  // radians
  static constexpr double kJacEps = 1e-6;  // degrees
  double p0 = 0, p1 = 0, lm = 1e-4;
  std::vector<double> r, ra, rb, r_try;
  residuals(p0, p1, r);
  const size_t n = r.size();
  std::vector<double> w(n, 1.0);
  auto weighted_sse = [&](const std::vector<double>& rr) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += w[i] * rr[i] * rr[i];
    return s;
  };
  for (int iter = 0; iter < 30; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      const double a = std::abs(r[i]);
      w[i] = a <= kHuber ? 1.0 : kHuber / a;
    }
    residuals(p0 + kJacEps, p1, ra);
    residuals(p0, p1 + kJacEps, rb);
    // Normal equations for 2x2 system
    double A00 = lm, A01 = 0, A11 = lm, g0 = 0, g1 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double j0 = (ra[i] - r[i]) / kJacEps;
      const double j1 = (rb[i] - r[i]) / kJacEps;
      A00 += w[i] * j0 * j0;
      A01 += w[i] * j0 * j1;
      A11 += w[i] * j1 * j1;
      g0 += w[i] * j0 * r[i];
      g1 += w[i] * j1 * r[i];
    }
    const double det = A00 * A11 - A01 * A01;
    if (std::abs(det) < 1e-18) break;
    const double s0 = (-g0 * A11 + g1 * A01) / det;
    const double s1 = (-g1 * A00 + g0 * A01) / det;
    residuals(p0 + s0, p1 + s1, r_try);
    if (weighted_sse(r_try) < weighted_sse(r)) {
      p0 += s0;
      p1 += s1;
      r = r_try;
      lm = std::max(lm * 0.5, 1e-9);
    } else {
      lm *= 10.0;
    }
    if (std::sqrt(s0 * s0 + s1 * s1) < 1e-8) break;
  }

  double rms1 = 0;
  for (const double rr : r) rms1 += rr * rr;
  rms1 = std::sqrt(rms1 / n);
  report.rms_after_px = rms1 * (180.0 / M_PI) * solver_eqr_size / 180.0;
  report.d_pitch_deg = p0;
  report.d_roll_deg = p1;
  calib.right.pitch += p0;
  calib.right.roll += p1;
  report.success = true;
  report.message = "OK";
  return report;
}

bool computeColorMatchGains(
    RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R)
{
  RigCalibration no_gain = calib;
  no_gain.gain_bgr[0] = no_gain.gain_bgr[1] = no_gain.gain_bgr[2] = 1.0;
  cv::Mat eq_L, eq_R;
  stitchEyes(no_gain, warps, frame_L, frame_R, eq_L, eq_R, cv::INTER_LINEAR);
  // Central region only (avoid vignette + out-of-coverage black)
  const int S = warps.eqr_size;
  const cv::Rect central(S / 5, S / 5, 3 * S / 5, 3 * S / 5);
  const cv::Scalar mean_L = cv::mean(eq_L(central));
  const cv::Scalar mean_R = cv::mean(eq_R(central));
  for (int c = 0; c < 3; ++c) {
    if (mean_L[c] < 1.0) return false;
    calib.gain_bgr[c] = math::clamp(mean_R[c] / mean_L[c], 0.7, 1.4);
  }
  return true;
}

void computeStartFrames(const RigCalibration& calib, int& start_L, int& start_R)
{
  const int skip = calib.skip_first_frame ? 1 : 0;
  start_L = std::max(0, -calib.sync_offset_frames) + skip;
  start_R = std::max(0, calib.sync_offset_frames) + skip;
}

namespace {

std::string pickProresEncoderArgs(const std::string& ffmpeg)
{
  const std::string encoders = execBlockingWithOutput(ffmpeg + " -hide_banner -encoders");
  if (encoders.find("prores_videotoolbox") != std::string::npos) {
    return "-c:v prores_videotoolbox -profile:v hq";  // hardware ProRes on Apple Silicon
  }
  return "-c:v prores_ks -profile:v 3 -vendor apl0";
}

}  // namespace

bool exportStitched(
    const RigCalibration& calib, const ExportConfig& cfg, std::string& error_message)
{
  cv::VideoCapture cap_L(cfg.left_path), cap_R(cfg.right_path);
  if (!cap_L.isOpened()) {
    error_message = "Failed to open left video: " + cfg.left_path;
    return false;
  }
  if (!cap_R.isOpened()) {
    error_message = "Failed to open right video: " + cfg.right_path;
    return false;
  }

  int start_L, start_R;
  computeStartFrames(calib, start_L, start_R);
  cap_L.set(cv::CAP_PROP_POS_FRAMES, start_L);
  cap_R.set(cv::CAP_PROP_POS_FRAMES, start_R);

  double fps = cfg.fps_override > 0 ? cfg.fps_override : cap_L.get(cv::CAP_PROP_FPS);
  if (fps <= 0 || fps > 240) fps = 29.97;
  const int total_guess =
      std::max(1, int(cap_L.get(cv::CAP_PROP_FRAME_COUNT)) - start_L);

  cv::Mat frame_L, frame_R;
  cap_L >> frame_L;
  cap_R >> frame_R;
  if (frame_L.empty() || frame_R.empty()) {
    error_message = "Could not decode first frame pair.";
    return false;
  }
  XPLINFO << "Input " << frame_L.cols << "x" << frame_L.rows << " @ " << fps
          << " fps; output " << cfg.eqr_size * 2 << "x" << cfg.eqr_size;

  StitchWarps warps;
  buildWarps(calib, frame_L.size(), cfg.eqr_size, warps);

  const bool png_mode = (cfg.codec == "png");
  FILE* pipe = nullptr;
  if (png_mode) {
    file::createDirectoryIfNotExists(cfg.out_path);
  } else {
    const std::string ext = file::filenameExtension(cfg.out_path);
    std::string vcodec, acodec;
    if (cfg.codec == "prores") {
      vcodec = pickProresEncoderArgs(cfg.ffmpeg);
      acodec = "-c:a pcm_s16le";
    } else if (cfg.codec == "h265") {
      vcodec = "-c:v libx265 -preset medium -crf 20 -pix_fmt yuv420p10le -tag:v hvc1 "
               "-movflags faststart";
      acodec = "-c:a aac -b:a 192k";
    } else {
      error_message = "Unknown codec: " + cfg.codec;
      return false;
    }
    std::string audio_in = "", audio_map = "";
    if (cfg.mux_audio) {
      const double audio_seek = start_R / fps;
      audio_in = " -ss " + std::to_string(audio_seek) + " -i \"" + cfg.right_path + "\"";
      audio_map = " -map 1:a:0? " + acodec;
    }
    const std::string cmd =
        cfg.ffmpeg + " -y -hide_banner -loglevel error -f rawvideo -pix_fmt bgr24 -s " +
        std::to_string(cfg.eqr_size * 2) + "x" + std::to_string(cfg.eqr_size) + " -r " +
        std::to_string(fps) + " -i -" + audio_in + " -map 0:v " + vcodec + audio_map +
        " \"" + cfg.out_path + "\"";
    XPLINFO << "ffmpeg command: " << cmd;
    pipe = K2_POPEN(cmd.c_str(), "w");
    if (!pipe) {
      error_message = "Failed to launch ffmpeg. Check the ffmpeg path in Settings.";
      return false;
    }
  }

  int frame_index = 0;
  int written = 0;
  bool cancelled = false;
  while (true) {
    if (cfg.cancel && *cfg.cancel) {
      cancelled = true;
      break;
    }
    if (frame_index > 0) {  // first pair already decoded above
      cap_L >> frame_L;
      cap_R >> frame_R;
      if (frame_L.empty() || frame_R.empty()) break;
    }
    const bool in_range = (cfg.first_frame < 0 || frame_index >= cfg.first_frame) &&
                          (cfg.last_frame < 0 || frame_index <= cfg.last_frame);
    if (in_range) {
      const cv::Mat sbs = stitchFrame(calib, warps, frame_L, frame_R, cv::INTER_CUBIC);
      XCHECK(sbs.isContinuous());
      if (png_mode) {
        cv::imwrite(
            cfg.out_path + "/sbs_" + string::intToZeroPad(written, 6) + ".png", sbs);
      } else {
        const size_t bytes = sbs.total() * sbs.elemSize();
        if (fwrite(sbs.data, 1, bytes, pipe) != bytes) {
          error_message = "ffmpeg pipe write failed (ffmpeg may have exited early).";
          K2_PCLOSE(pipe);
          return false;
        }
      }
      ++written;
      if (cfg.progress) cfg.progress(written, total_guess);
    }
    if (cfg.last_frame >= 0 && frame_index >= cfg.last_frame) break;
    ++frame_index;
  }

  if (pipe) {
    const int rc = K2_PCLOSE(pipe);
    if (!cancelled && rc != 0) {
      error_message = "ffmpeg exited with code " + std::to_string(rc);
      return false;
    }
  }
  if (cancelled) {
    error_message = "Export cancelled. Partial output may exist.";
    return false;
  }
  XPLINFO << "Export complete: " << written << " frames -> " << cfg.out_path;
  return true;
}

cv::Mat stitchStillFromVideos(
    const RigCalibration& calib,
    const std::string& left_path,
    const std::string& right_path,
    const int frame_index,
    const int eqr_size)
{
  cv::VideoCapture cap_L(left_path), cap_R(right_path);
  if (!cap_L.isOpened() || !cap_R.isOpened()) return cv::Mat();
  int start_L, start_R;
  computeStartFrames(calib, start_L, start_R);
  cap_L.set(cv::CAP_PROP_POS_FRAMES, start_L + frame_index);
  cap_R.set(cv::CAP_PROP_POS_FRAMES, start_R + frame_index);
  cv::Mat frame_L, frame_R;
  cap_L >> frame_L;
  cap_R >> frame_R;
  if (frame_L.empty() || frame_R.empty()) return cv::Mat();
  StitchWarps warps;
  buildWarps(calib, frame_L.size(), eqr_size, warps);
  return stitchFrame(calib, warps, frame_L, frame_R, cv::INTER_CUBIC);
}

}}  // namespace p11::k2stitch
