// MIT License. Headless CLI driver for the VR180 -> LDI3 video pipeline.
// Writes jg4d_sidecar.json next to the output for the jg4d Blender player.
// Built/validated on Linux CUDA (RunPod RTX 4090), 2026-07-17.
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <gflags/gflags.h>
#include "logger.h"
#include "ldi_pipeline_lib.h"

DEFINE_string(src_vr180, "", "Source VR180 stereo video. Required for video mode.");
DEFINE_string(src_ftheta_image, "", "Photo-mode input: f-theta image.");
DEFINE_string(src_ftheta_depth, "", "Photo-mode: matching f-theta depthmap.");
DEFINE_string(dest_dir, "", "Output/working directory. Required.");
DEFINE_string(output_filename, "ldi3_000000.png", "Output name (photo mode). GUI default.");
DEFINE_string(cwd, "", "Working directory override.");
DEFINE_bool(rm_dest_dir, false, "Delete dest_dir before starting.");
DEFINE_string(
    output_encoding,
    "split12",
    "LDI3 grid encoding: split12 (default; 12-bit depth, what the jg4d Blender player "
    "and the web player decode) | 8bit | 16bit. WARNING: make6DofGrid only scales pixel "
    "data for these three values - any other value leaves float 0-1 data and writes "
    "BLACK frames.");
DEFINE_bool(photo_mode, false, "Run the still-photo pipeline.");
DEFINE_int32(
    ftheta_size,
    3840,
    "WORKING f-theta resolution (GUI default 3840). This is NOT the output size: "
    "layers are computed at this resolution and then warped down to "
    "inflated_ftheta_size, so ftheta_size > inflated_ftheta_size is supersampling. "
    "Raise this to improve quality at a fixed output size.");
DEFINE_int32(
    inflated_ftheta_size,
    1920,
    "OUTPUT resolution per grid tile (GUI default 1920). The LDI3 grid is 3x3 tiles "
    "of this size, so the written frame is 3x this value: 1920 -> 5760x5760, "
    "2048 -> 6144x6144. Raise this to increase output resolution. "
    "0 = fall back to ftheta_size; must be > 0 by the time it reaches the pipeline, "
    "since a 0-size warp camera produces an empty warp and BLACK frames.");
DEFINE_int32(rectified_size_for_depth, 1280,
    "Rectified pair size for stereo depth (GUI default 1280).");
DEFINE_double(disparity_bias, 0.0, "disparity bias.");
DEFINE_double(baseline_m, 0.060, "stereo baseline meters (R5C RF5.2mm = 0.060).");
DEFINE_double(inv_depth_coef, 0.3, "inverse depth coefficient.");
DEFINE_double(ftheta_scale, 1.15, "f-theta scale.");
DEFINE_string(depth_method, "raft", "raft | da3_fused | da3_only");
DEFINE_double(da3_blend, 0.6, "0=stereo, 1=aligned DA3.");
DEFINE_string(da3_model_path, "", "Override path to da3_stereo.pt.");
// These four default to the GUI's values. Empty/zero are NOT harmless: an empty
// inpaint_method matches neither branch in ldi_common.cc so inpainting is silently
// skipped, an empty seg_method likewise skips segmentation, and a dilate radius of
// 0 makes cv::dilate a no-op AND collapses the alpha blend kernel to 1x1, giving
// hard-edged layer seams instead of feathered ones.
DEFINE_string(inpaint_method, "ceres", "Inpaint method (GUI default ceres).");
DEFINE_string(seg_method, "heuristic", "Segmentation method (GUI default heuristic).");
DEFINE_string(sd_ver, "v2", "SD version tag (GUI default v2; SD path is disabled).");
DEFINE_int32(inpaint_dilate_radius, 25,
    "Inpaint mask dilation + alpha feather radius (GUI default 25).");
DEFINE_bool(stabilize_inpainting, false, "run stabilizeInpaintingPhase.");
DEFINE_bool(run_seg_only, false, "only run segmentation.");
DEFINE_bool(write_seg, false, "write segmentation images.");
DEFINE_bool(use_cached_seg, false, "reuse cached segmentation.");
DEFINE_bool(make_fused_image, false, "write fused debug image.");
DEFINE_int32(first_frame, 0, "first frame.");
DEFINE_int32(last_frame, -1, "last frame (-1 = end).");
DEFINE_bool(skip_every_other_frame, false, "half-rate.");
DEFINE_string(phase, "all", "all|depth|stabilize|inpaint|stabilize_inpainting|cleanup");

// Static-camera background plates. These were previously reachable only through
// the LIFECAST_PLATE_PATH / LIFECAST_DEPTH_PLATE_PATH environment variables,
// which appeared in no --help output and so were effectively undiscoverable.
// The flags below set those variables, so ldi_common.cc keeps its existing
// interface and the env vars still work for other callers.
DEFINE_string(
    plate_path, "",
    "Colour background plate (PNG) for a locked-off shot. Applied ONLY inside "
    "layer-0 disocclusion holes, so live motion (grass, foliage) is preserved. "
    "Build one with scripts/build_plates.py. Overrides $LIFECAST_PLATE_PATH.");
DEFINE_string(
    depth_plate_path, "",
    "Depth background plate (PNG) for a locked-off shot. Freezes background "
    "geometry to remove per-frame depth wobble. "
    "Overrides $LIFECAST_DEPTH_PLATE_PATH.");

namespace {
void writeJg4dSidecar(const p11::ldi::LdiPipelineConfig& cfg) {
    const std::string path = cfg.dest_dir + "/jg4d_sidecar.json";
  std::ofstream f(path);
  if (!f) { XPLINFO << "WARNING: could not write sidecar: " << path; return; }
  f << "{\n"
        << "  \"inv_depth_coef\": " << cfg.inv_depth_coef << ",\n"
        << "  \"ftheta_scale\": " << cfg.ftheta_scale << ",\n"
        << "  \"baseline_m\": " << cfg.baseline_m << ",\n"
        << "  \"depth_method\": \"" << cfg.depth_method << "\",\n"
        << "  \"da3_blend\": " << cfg.da3_blend << ",\n"
        << "  \"decode_12bit\": true\n" << "}\n";
  XPLINFO << "Wrote " << path;
}
}  // namespace

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_dest_dir.empty()) { XPLINFO << "--dest_dir is required"; return 1; }
  if (!FLAGS_photo_mode && FLAGS_src_vr180.empty() && FLAGS_src_ftheta_image.empty()) {
    XPLINFO << "--src_vr180 or --src_ftheta_image required"; return 1; }
  // Plates travel to ldi_common.cc through the environment; setting them here
  // means they show up in --help and are logged, instead of being an undocumented
  // env-var side channel.
  if (!FLAGS_plate_path.empty()) {
    setenv("LIFECAST_PLATE_PATH", FLAGS_plate_path.c_str(), /*overwrite=*/1);
    XPLINFO << "colour plate: " << FLAGS_plate_path;
  }
  if (!FLAGS_depth_plate_path.empty()) {
    setenv("LIFECAST_DEPTH_PLATE_PATH", FLAGS_depth_plate_path.c_str(), /*overwrite=*/1);
    XPLINFO << "depth plate: " << FLAGS_depth_plate_path;
  }

  p11::ldi::LdiPipelineConfig cfg;
  cfg.cancel_requested = std::make_shared<std::atomic<bool>>(false);
  cfg.cwd = FLAGS_cwd; cfg.src_vr180 = FLAGS_src_vr180;
  cfg.src_ftheta_image = FLAGS_src_ftheta_image; cfg.src_ftheta_depth = FLAGS_src_ftheta_depth;
  cfg.dest_dir = FLAGS_dest_dir; cfg.output_filename = FLAGS_output_filename;
  cfg.rm_dest_dir = FLAGS_rm_dest_dir; cfg.ftheta_size = FLAGS_ftheta_size;
  // Fall back to ftheta_size: a 0 inflated size builds a 0-pixel warp camera, which
  // silently yields an empty warp and black LDI3 frames.
  cfg.inflated_ftheta_size =
      FLAGS_inflated_ftheta_size > 0 ? FLAGS_inflated_ftheta_size : FLAGS_ftheta_size;
  cfg.rectified_size_for_depth = FLAGS_rectified_size_for_depth;
  cfg.disparity_bias = FLAGS_disparity_bias; cfg.baseline_m = FLAGS_baseline_m;
  cfg.inv_depth_coef = FLAGS_inv_depth_coef; cfg.ftheta_scale = FLAGS_ftheta_scale;
  cfg.inpaint_method = FLAGS_inpaint_method; cfg.seg_method = FLAGS_seg_method;
  cfg.sd_ver = FLAGS_sd_ver; cfg.first_frame = FLAGS_first_frame; cfg.last_frame = FLAGS_last_frame;
  cfg.phase = FLAGS_phase; cfg.stabilize_inpainting = FLAGS_stabilize_inpainting;
  cfg.run_seg_only = FLAGS_run_seg_only; cfg.write_seg = FLAGS_write_seg;
  cfg.use_cached_seg = FLAGS_use_cached_seg; cfg.output_encoding = FLAGS_output_encoding;
  cfg.make_fused_image = FLAGS_make_fused_image;
  cfg.inpaint_dilate_radius = FLAGS_inpaint_dilate_radius;
  cfg.skip_every_other_frame = FLAGS_skip_every_other_frame;
  cfg.depth_method = FLAGS_depth_method; cfg.da3_blend = FLAGS_da3_blend;
  cfg.da3_model_path = FLAGS_da3_model_path;
  if (cfg.output_encoding != "split12" && cfg.output_encoding != "8bit" &&
      cfg.output_encoding != "16bit") {
    XPLINFO << "FATAL: --output_encoding=" << cfg.output_encoding
            << " would produce black frames. Use split12 (default), 8bit, or 16bit.";
    return 1;
  }
  // The grid is 3x3 tiles of the INFLATED size (make6DofGrid warps every layer
  // through warp_ftheta_to_inflated, whose output is sized to cam_inflated).
  // Reporting 3 * ftheta_size here would be wrong whenever the two differ.
  XPLINFO << "working f-theta resolution: " << cfg.ftheta_size
          << " -> LDI3 grid " << (3 * cfg.inflated_ftheta_size) << "x"
          << (3 * cfg.inflated_ftheta_size)
          << " (tile " << cfg.inflated_ftheta_size << ", supersample "
          << (double(cfg.ftheta_size) / double(cfg.inflated_ftheta_size)) << "x)";
  p11::ldi::printConfig(cfg);
  if (FLAGS_photo_mode) { p11::ldi::runVR180PhototoLdiPipeline(cfg); }
  else if (FLAGS_phase == "all") { p11::ldi::runVR180toLdi3VideoPipelineAllPhases(cfg); }
  else if (FLAGS_phase == "depth") { p11::ldi::videoDepthPhase(cfg); }
  else if (FLAGS_phase == "stabilize") { p11::ldi::temporallyStabilizeDepth(cfg); }
  else if (FLAGS_phase == "inpaint") { p11::ldi::inpaintPhase(cfg); }
  else if (FLAGS_phase == "stabilize_inpainting") { p11::ldi::stabilizeInpaintingPhase(cfg); }
  else if (FLAGS_phase == "cleanup") { p11::ldi::cleanup(cfg); }
  else { XPLINFO << "Unknown --phase: " << FLAGS_phase; return 1; }
  writeJg4dSidecar(cfg);
  XPLINFO << "vve_cli done.";
  return 0;
}
