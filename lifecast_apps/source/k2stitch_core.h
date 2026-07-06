// MIT License. k2stitch core: warp construction, stitching, auto-refine solver,
// color match, and ffmpeg export. GUI-independent (used by both the k2stitch GUI
// and its --batch CLI mode).
//
// The warp/convention math is validated end-to-end against Lifecast's own
// projection code by k2stitch_test_math.py.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "fisheye_camera.h"
#include "k2stitch_calibration.h"

namespace p11 { namespace k2stitch {

// Precomputed remap tables (equirect output pixel -> raw fisheye source pixel).
struct StitchWarps {
  std::vector<cv::Mat> warp_L, warp_R;  // each: {map_x, map_y}, CV_32F
  int eqr_size = 0;                     // per-eye output size (square)
  cv::Size video_size;                  // raw fisheye frame size the maps were built for
};

// Equivalent to projection::precomputeremapFisheyeToEquirectWarp (the "180 equirect"
// warp), plus explicit invalidation of rays beyond the lens coverage. The output
// convention is exactly what VVE's precomputeVR180toFthetaWarp samples (validated
// in k2stitch_test_math.py T2).
void precomputeFisheyeToVR180Warp(
    const calibration::FisheyeCamerad& cam,
    const int eqr_size,
    const double lens_fov_deg,
    std::vector<cv::Mat>& warp_uv);

// Build both eyes' warps from the rig calibration.
void buildWarps(
    const RigCalibration& calib,
    const cv::Size& video_size,
    const int eqr_size,
    StitchWarps& warps);

// Remap both raw fisheye frames into a side-by-side VR180 equirect frame
// (left half = left eye). Applies calib.gain_bgr to the left (slave) eye and
// handles calib.swap_eyes.
cv::Mat stitchFrame(
    const RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    const cv::InterpolationFlags interp = cv::INTER_CUBIC);

// Stitch each eye separately (for preview compositing).
void stitchEyes(
    const RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    cv::Mat& eq_L,
    cv::Mat& eq_R,
    const cv::InterpolationFlags interp = cv::INTER_LINEAR);

// Preview composites. Inputs are the two stitched eyes (same size, CV_8UC3).
cv::Mat compositeAnaglyph(const cv::Mat& eq_L, const cv::Mat& eq_R);   // red=L, cyan=R
cv::Mat compositeBlend(const cv::Mat& eq_L, const cv::Mat& eq_R);      // 50/50
cv::Mat compositeDifference(const cv::Mat& eq_L, const cv::Mat& eq_R); // abs diff x4

// Draw horizontal epipolar gridlines (constant vertical angle) every spacing_deg.
// Vertical misalignment shows as red/cyan separation across these lines.
void drawLatitudeGrid(cv::Mat& eye_or_composite, const int spacing_deg = 5);

struct RefineReport {
  bool success = false;
  int num_matches = 0;
  double rms_before_px = 0;  // RMS vertical disparity at eqr_size px/eye
  double rms_after_px = 0;
  double d_pitch_deg = 0;  // applied to calib.right.pitch
  double d_roll_deg = 0;   // applied to calib.right.roll
  std::string message;
};

// Auto-refine: ORB feature matching between the two stitched eyes, then
// Levenberg-Marquardt (IRLS-Huber) on the right eye's (pitch, roll), minimizing
// vertical angular disparity. Horizontal disparity is scene depth and is NOT
// penalized. On success, calib.right.pitch/roll are updated.
// Validated in k2stitch_test_math.py T3 (recovers injected misalignment to <0.01 deg).
RefineReport refineAlignment(
    const cv::Mat& frame_L,
    const cv::Mat& frame_R,
    RigCalibration& calib,
    const int solver_eqr_size = 1536);

// Estimate per-channel gains that match the left (slave) eye to the right (master),
// using means over the central stitched region. Writes calib.gain_bgr.
bool computeColorMatchGains(
    RigCalibration& calib,
    const StitchWarps& warps,
    const cv::Mat& frame_L,
    const cv::Mat& frame_R);

struct ExportConfig {
  std::string left_path, right_path;  // input videos
  std::string out_path;               // .mov/.mp4 path, or directory for codec=="png"
  std::string ffmpeg = "ffmpeg";      // ffmpeg binary
  int eqr_size = 2880;                // per-eye output resolution
  std::string codec = "prores";       // "prores" | "h265" | "png"
  int first_frame = -1;               // -1 = from start (after sync/skip adjustments)
  int last_frame = -1;                // -1 = to end
  double fps_override = 0;            // 0 = take from left video
  bool mux_audio = true;              // audio from the right (master) file
  std::shared_ptr<std::atomic<bool>> cancel;
  std::function<void(int frame, int total)> progress;
};

// Returns true on success; on failure error_message explains.
bool exportStitched(
    const RigCalibration& calib, const ExportConfig& cfg, std::string& error_message);

// Stitch a single frame pair from the input videos (sync offset + skip applied).
// Returns an empty Mat on failure.
cv::Mat stitchStillFromVideos(
    const RigCalibration& calib,
    const std::string& left_path,
    const std::string& right_path,
    const int frame_index,
    const int eqr_size);

// First frame index of the LEFT video after sync offset + skip_first_frame.
void computeStartFrames(const RigCalibration& calib, int& start_L, int& start_R);

}}  // namespace p11::k2stitch
