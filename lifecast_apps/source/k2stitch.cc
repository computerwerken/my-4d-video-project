// MIT License. k2stitch: align & stitch Z CAM K2 Pro dual-fisheye footage into the
// side-by-side VR180 equirect format that Lifecast Volumetric Video Editor ingests
// (the role EOS VR Utility plays for Canon R5/R5C dual-fisheye footage).
//
// GUI mode:   k2stitch
// Batch mode: k2stitch --batch --left L.mov --right R.mov --calib rig.json \
//                      --out sbs.mov [--eqr_size 2880] [--codec prores|h265|png]
// Still mode: k2stitch --still 0 --left L.mov --right R.mov --calib rig.json --out sbs.png
//
// Handoff to VVE: File -> Import VR180, Render LDI3 with the exported file,
// baseline = 6.4 cm (K2 Pro IPD is 64 mm; VVE's default is 6.0).

// Make the application run without a terminal in Windows.
#if defined(windows_hide_console) && defined(_WIN32)
#pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
#endif

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "gflags/gflags.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

#include "dear_imgui_app.h"
#include "imgui_cvmat.h"
#include "imgui_filedialog.h"
#include "logger.h"
#include "preferences.h"
#include "util_command.h"
#include "util_file.h"
#include "util_math.h"

#include "k2stitch_calibration.h"
#include "k2stitch_core.h"

DEFINE_bool(batch, false, "Run headless export instead of the GUI");
DEFINE_int32(still, -1, "Stitch a single frame index to --out (PNG) and exit");
DEFINE_string(left, "", "Left eye (slave unit) video path");
DEFINE_string(right, "", "Right eye (master unit) video path");
DEFINE_string(calib, "", "Rig calibration JSON (see k2stitch_calibration.h)");
DEFINE_string(out, "", "Output path (.mov/.mp4, directory for --codec png, .png for --still)");
DEFINE_int32(eqr_size, 2880, "Per-eye output resolution (square)");
DEFINE_string(codec, "prores", "prores | h265 | png");
DEFINE_int32(first_frame, -1, "First frame to export (-1 = start)");
DEFINE_int32(last_frame, -1, "Last frame to export (-1 = end)");
DEFINE_double(fps, 0, "Override output fps (0 = from left video)");
DEFINE_string(ffmpeg, "", "Path to ffmpeg (default: /usr/local/bin/ffmpeg on macOS, else ffmpeg)");

namespace p11 {

constexpr float kK2StitchVersion = 0.1;

struct K2StitchApp : public DearImGuiApp {
  float kToolbarWidth = 360;
  int menu_bar_height = 20;

  std::map<std::string, std::string> prefs;
  std::string ffmpeg;
  gui::ImguiInputFileSelect left_file_select;
  gui::ImguiInputFileSelect right_file_select;
  gui::ImguiInputFileSelect ffmpeg_tool_select;

  k2stitch::RigCalibration calib;
  std::mutex state_mutex;  // guards calib + raw frames for the preview thread

  cv::VideoCapture cap_L, cap_R;
  cv::Mat raw_L, raw_R;
  bool have_video = false;
  int num_frames = 0;
  int curr_frame = 0;
  double fps = 29.97;
  std::string video_info = "No videos loaded.";

  enum PreviewMode { kAnaglyph = 0, kSBS, kBlend, kDifference, kLeftOnly, kRightOnly };
  int preview_mode = kAnaglyph;
  bool show_grid = true;
  int preview_size = 768;

  std::atomic<bool> preview_dirty = false;
  std::atomic<bool> preview_ready = false;
  std::atomic<bool> shutting_down = false;
  cv::Mat preview_pending;
  std::mutex preview_mutex;
  gui::ImguiCvMat preview_image;
  std::shared_ptr<std::thread> preview_thread;

  CommandRunner command_runner;
  std::shared_ptr<std::atomic<bool>> cancel_requested =
      std::make_shared<std::atomic<bool>>(false);
  std::atomic<int> job_progress_frame = 0;
  std::atomic<int> job_progress_total = 1;
  std::mutex refine_status_mutex;
  std::string refine_status = "";

  void setRefineStatus(const std::string& s)
  {
    std::lock_guard<std::mutex> lock(refine_status_mutex);
    refine_status = s;
  }
  std::string getRefineStatus()
  {
    std::lock_guard<std::mutex> lock(refine_status_mutex);
    return refine_status;
  }

  bool open_popup_export = false;
  char export_path_buf[1024] = "";
  int export_eqr_choice = 1;    // 0: 2048, 1: 2880, 2: 3840
  int export_codec_choice = 0;  // 0: prores, 1: h265, 2: png

  ~K2StitchApp() { preview_image.freeGlTexture(); }

  void setExternalToolDefaultPaths()
  {
#ifdef __APPLE__
    ffmpeg = "/usr/local/bin/ffmpeg";
#else
    ffmpeg = "ffmpeg";
#endif
  }

  void initApp()
  {
    prefs = preferences::getPrefs();
    setExternalToolDefaultPaths();
    if (prefs.count("k2_ffmpeg")) ffmpeg = prefs.at("k2_ffmpeg");
    if (!FLAGS_ffmpeg.empty()) ffmpeg = FLAGS_ffmpeg;
    ffmpeg_tool_select.setPath(ffmpeg.c_str());
    if (prefs.count("k2_left")) left_file_select.setPath(prefs.at("k2_left").c_str());
    if (prefs.count("k2_right")) right_file_select.setPath(prefs.at("k2_right").c_str());
    left_file_select.label = "Left eye video (slave unit)";
    right_file_select.label = "Right eye video (master unit)";
    ffmpeg_tool_select.label = "ffmpeg";
    if (!FLAGS_left.empty()) left_file_select.setPath(FLAGS_left.c_str());
    if (!FLAGS_right.empty()) right_file_select.setPath(FLAGS_right.c_str());
    startPreviewThread();
  }

  void markPreviewDirty() { preview_dirty = true; }

  void startPreviewThread()
  {
    preview_thread = std::make_shared<std::thread>([&] {
      while (!shutting_down) {
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        if (!preview_dirty || !have_video) continue;
        preview_dirty = false;

        k2stitch::RigCalibration c;
        cv::Mat L, R;
        int mode, psize;
        bool grid;
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          c = calib;
          L = raw_L.clone();
          R = raw_R.clone();
          mode = preview_mode;
          psize = preview_size;
          grid = show_grid;
        }
        if (L.empty() || R.empty()) continue;

        k2stitch::StitchWarps warps;
        k2stitch::buildWarps(c, L.size(), psize, warps);
        cv::Mat eq_L, eq_R;
        k2stitch::stitchEyes(c, warps, L, R, eq_L, eq_R, cv::INTER_LINEAR);

        cv::Mat composite;
        switch (mode) {
          case kSBS:
            cv::hconcat(eq_L, eq_R, composite);
            break;
          case kAnaglyph:
            composite = k2stitch::compositeAnaglyph(eq_L, eq_R);
            break;
          case kBlend:
            composite = k2stitch::compositeBlend(eq_L, eq_R);
            break;
          case kDifference:
            composite = k2stitch::compositeDifference(eq_L, eq_R);
            break;
          case kLeftOnly:
            composite = eq_L;
            break;
          case kRightOnly:
            composite = eq_R;
            break;
          default:
            composite = eq_L;
        }
        if (grid) k2stitch::drawLatitudeGrid(composite);

        {
          std::lock_guard<std::mutex> lock(preview_mutex);
          preview_pending = composite;
        }
        preview_ready = true;
      }
    });
    preview_thread->detach();
  }

  void loadVideos()
  {
    const std::string path_L(left_file_select.path);
    const std::string path_R(right_file_select.path);
    if (!file::fileExists(path_L) || !file::fileExists(path_R)) {
      tinyfd_messageBox(
          "File Not Found", "Select both left and right eye videos first.", "ok", "error", 1);
      return;
    }
    cap_L.open(path_L);
    cap_R.open(path_R);
    if (!cap_L.isOpened() || !cap_R.isOpened()) {
      tinyfd_messageBox(
          "Could Not Open Video",
          "OpenCV/FFmpeg failed to decode one of the inputs.",
          "ok", "error", 1);
      return;
    }
    fps = cap_L.get(cv::CAP_PROP_FPS);
    const int w = int(cap_L.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = int(cap_L.get(cv::CAP_PROP_FRAME_HEIGHT));
    int start_L, start_R;
    k2stitch::computeStartFrames(calib, start_L, start_R);
    num_frames = std::max(
        1,
        std::min(
            int(cap_L.get(cv::CAP_PROP_FRAME_COUNT)) - start_L,
            int(cap_R.get(cv::CAP_PROP_FRAME_COUNT)) - start_R));
    video_info = std::to_string(w) + "x" + std::to_string(h) + " @ " +
                 std::to_string(fps).substr(0, 5) + " fps, " + std::to_string(num_frames) +
                 " frames";
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      // Sensible defaults for a fresh calibration (MKX200 circle ~ fills sensor height)
      if (calib.left.center_x <= 0) calib.left.center_x = w / 2.0;
      if (calib.left.center_y <= 0) calib.left.center_y = h / 2.0;
      if (calib.right.center_x <= 0) calib.right.center_x = w / 2.0;
      if (calib.right.center_y <= 0) calib.right.center_y = h / 2.0;
      if (calib.left.circle_radius <= 0) calib.left.circle_radius = std::min(w, h) * 0.4875;
      if (calib.right.circle_radius <= 0) calib.right.circle_radius = std::min(w, h) * 0.4875;
    }
    have_video = true;
    curr_frame = 0;
    seekAndReadFrame(0);
    prefs["k2_left"] = path_L;
    prefs["k2_right"] = path_R;
    preferences::setPrefs(prefs);
  }

  void seekAndReadFrame(const int f)
  {
    if (!have_video) return;
    int start_L, start_R;
    k2stitch::computeStartFrames(calib, start_L, start_R);
    cap_L.set(cv::CAP_PROP_POS_FRAMES, start_L + f);
    cap_R.set(cv::CAP_PROP_POS_FRAMES, start_R + f);
    cv::Mat L, R;
    cap_L >> L;
    cap_R >> R;
    if (L.empty() || R.empty()) return;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      raw_L = L;
      raw_R = R;
    }
    markPreviewDirty();
  }

  // ---- menu handlers ----

  void handleMenuLoadCalibration()
  {
    const char* cpath = tinyfd_openFileDialog(
        "Load rig calibration (.json)", nullptr, 0, nullptr, nullptr, 0);
    if (cpath == nullptr) return;
    k2stitch::RigCalibration loaded;
    if (!k2stitch::loadCalibration(cpath, loaded)) {
      tinyfd_messageBox("Load Failed", "Could not parse calibration JSON.", "ok", "error", 1);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      calib = loaded;
    }
    markPreviewDirty();
  }

  void handleMenuSaveCalibration()
  {
    const char* cpath = tinyfd_saveFileDialog(
        "Save rig calibration (.json)", "k2_rig_calibration.json", 0, nullptr, nullptr);
    if (cpath == nullptr) return;
    k2stitch::RigCalibration snapshot;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      snapshot = calib;
    }
    if (!k2stitch::saveCalibration(snapshot, cpath)) {
      tinyfd_messageBox("Save Failed", "Could not write calibration JSON.", "ok", "error", 1);
    }
  }

  void handleMenuExportStill()
  {
    if (!have_video) return;
    const char* cpath = tinyfd_saveFileDialog(
        "Save current frame as SBS VR180 (.png)", "sbs_vr180.png", 0, nullptr, nullptr);
    if (cpath == nullptr) return;
    k2stitch::RigCalibration c;
    cv::Mat L, R;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      c = calib;
      L = raw_L.clone();
      R = raw_R.clone();
    }
    k2stitch::StitchWarps warps;
    k2stitch::buildWarps(c, L.size(), FLAGS_eqr_size, warps);
    cv::imwrite(cpath, k2stitch::stitchFrame(c, warps, L, R, cv::INTER_CUBIC));
    tinyfd_messageBox(
        "Frame Exported",
        "SBS VR180 still saved. Test it in VVE with File -> Import VR180, Render LDI3 "
        "(set baseline to 6.4 cm for the K2 Pro).",
        "ok", "info", 1);
  }

  void runAutoRefine()
  {
    if (!have_video || command_runner.isRunning()) return;
    *cancel_requested = false;
    setRefineStatus("Refining...");
    command_runner.setCompleteOrKilledCallback([] {});
    command_runner.queueThreadCommand(cancel_requested, [&] {
      k2stitch::RigCalibration c;
      cv::Mat L, R;
      {
        std::lock_guard<std::mutex> lock(state_mutex);
        c = calib;
        L = raw_L.clone();
        R = raw_R.clone();
      }
      const k2stitch::RefineReport rep = k2stitch::refineAlignment(L, R, c);
      if (rep.success) {
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          calib.right.pitch = c.right.pitch;
          calib.right.roll = c.right.roll;
        }
        char buf[256];
        snprintf(
            buf, sizeof(buf),
            "%d matches | vert. disparity %.2f -> %.2f px | d_pitch %+.3f d_roll %+.3f deg",
            rep.num_matches, rep.rms_before_px, rep.rms_after_px, rep.d_pitch_deg,
            rep.d_roll_deg);
        setRefineStatus(buf);
        markPreviewDirty();
      } else {
        setRefineStatus("Refine failed: " + rep.message);
      }
    });
    command_runner.runCommandQueue();
  }

  void runColorMatch()
  {
    if (!have_video) return;
    k2stitch::RigCalibration c;
    cv::Mat L, R;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      c = calib;
      L = raw_L.clone();
      R = raw_R.clone();
    }
    k2stitch::StitchWarps warps;
    k2stitch::buildWarps(c, L.size(), 512, warps);
    if (k2stitch::computeColorMatchGains(c, warps, L, R)) {
      std::lock_guard<std::mutex> lock(state_mutex);
      for (int i = 0; i < 3; ++i) calib.gain_bgr[i] = c.gain_bgr[i];
      markPreviewDirty();
    }
  }

  void runExport()
  {
    if (!have_video || command_runner.isRunning()) return;
    *cancel_requested = false;
    job_progress_frame = 0;

    k2stitch::ExportConfig cfg;
    cfg.left_path = std::string(left_file_select.path);
    cfg.right_path = std::string(right_file_select.path);
    cfg.out_path = std::string(export_path_buf);
    cfg.ffmpeg = ffmpeg;
    cfg.eqr_size = export_eqr_choice == 0 ? 2048 : export_eqr_choice == 1 ? 2880 : 3840;
    cfg.codec = export_codec_choice == 0 ? "prores" : export_codec_choice == 1 ? "h265" : "png";
    cfg.cancel = cancel_requested;
    cfg.progress = [&](int frame, int total) {
      job_progress_frame = frame;
      job_progress_total = std::max(1, total);
    };

    command_runner.setCompleteCallback([] {
      tinyfd_messageBox(
          "Export Finished",
          "Next: open VVE, select a project directory, then File -> Import VR180, Render "
          "LDI3 with this file. Set baseline to 6.4 cm (K2 Pro IPD).",
          "ok", "info", 1);
    });
    command_runner.setKilledCallback([] {
      tinyfd_messageBox("Export Cancelled", "Partial output may exist.", "ok", "warn", 1);
    });
    k2stitch::RigCalibration snapshot;
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      snapshot = calib;
    }
    command_runner.queueThreadCommand(cancel_requested, [this, cfg, snapshot] {
      std::string err;
      if (!k2stitch::exportStitched(snapshot, cfg, err)) {
        XPLINFO << "Export failed: " << err;
      }
    });
    command_runner.runCommandQueue();
  }

  // ---- GUI drawing ----

  bool main_menu_is_hovered = false;
  void drawMainMenu()
  {
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Load Videos (from paths below)")) loadVideos();
        ImGui::Separator();
        if (ImGui::MenuItem("Load Calibration...")) handleMenuLoadCalibration();
        if (ImGui::MenuItem("Save Calibration...")) handleMenuSaveCalibration();
        ImGui::Separator();
        if (ImGui::MenuItem("Export Current Frame as SBS PNG...")) handleMenuExportStill();
        if (ImGui::MenuItem("Export Stitched Video...")) open_popup_export = true;
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Settings")) {
        if (ImGui::MenuItem("About K2Stitch")) {
          const std::string msg =
              "K2Stitch v" + std::to_string(kK2StitchVersion).substr(0, 3) +
              "\nDual-fisheye (Z CAM K2 Pro) -> VR180 SBS equirect for Lifecast VVE." +
              "\nOutput convention validated against VVE's ingest warp.";
          tinyfd_messageBox("K2Stitch", msg.c_str(), "ok", "info", 1);
        }
        ImGui::Separator();
        ImGui::Text("ffmpeg path:");
        ffmpeg_tool_select.drawAndUpdate();
        if (ImGui::MenuItem("Apply ffmpeg path")) {
          ffmpeg = std::string(ffmpeg_tool_select.path);
          prefs["k2_ffmpeg"] = ffmpeg;
          preferences::setPrefs(prefs);
        }
        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("How to hand off to VVE")) {
          tinyfd_messageBox(
              "K2Stitch -> VVE",
              "1. Export Stitched Video (ProRes recommended).\n"
              "2. In VVE: File -> Select Project Directory.\n"
              "3. File -> Import VR180, Render LDI3 -> choose the exported file.\n"
              "4. Set baseline to 6.4 cm (K2 Pro IPD is 64 mm).\n"
              "5. Edit layers / encode as usual in VVE.",
              "ok", "info", 1);
        }
        ImGui::EndMenu();
      }
      main_menu_is_hovered = !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
      ImGui::EndMenuBar();
    }
  }

  bool sliderAngle(const char* label, double* value, float lo, float hi)
  {
    float v = float(*value);
    const bool changed = ImGui::SliderFloat(label, &v, lo, hi, "%.3f deg");
    if (changed) *value = double(v);
    return changed;
  }

  bool dragValue(const char* label, double* value, float speed, float lo, float hi,
                 const char* fmt = "%.2f")
  {
    float v = float(*value);
    const bool changed = ImGui::DragFloat(label, &v, speed, lo, hi, fmt);
    if (changed) *value = double(v);
    return changed;
  }

  void drawEyePanel(const char* tag, k2stitch::EyeCalibration& eye)
  {
    bool dirty = false;
    ImGui::PushID(tag);
    ImGui::Text("%s", tag);
    dirty |= dragValue("Center X", &eye.center_x, 0.25f, 0.0f, 8192.0f);
    dirty |= dragValue("Center Y", &eye.center_y, 0.25f, 0.0f, 8192.0f);
    dirty |= dragValue("Circle radius", &eye.circle_radius, 0.25f, 100.0f, 4096.0f);
    dirty |= dragValue("Lens FOV", &eye.lens_fov_deg, 0.05f, 160.0f, 220.0f, "%.1f deg");
    dirty |= dragValue("k1", &eye.k1, 0.0005f, -0.2f, 0.2f, "%.4f");
    dirty |= dragValue("tilt", &eye.tilt, 0.0001f, -0.05f, 0.05f, "%.4f");
    dirty |= sliderAngle("Yaw", &eye.yaw, -3.0f, 3.0f);
    dirty |= sliderAngle("Pitch", &eye.pitch, -3.0f, 3.0f);
    dirty |= sliderAngle("Roll", &eye.roll, -3.0f, 3.0f);
    ImGui::PopID();
    if (dirty) markPreviewDirty();
  }

  void drawToolbar()
  {
    ImGui::BeginChild("##Toolbar", ImVec2(kToolbarWidth, 0), true);

    if (ImGui::CollapsingHeader("Ingest", ImGuiTreeNodeFlags_DefaultOpen)) {
      left_file_select.drawAndUpdate();
      right_file_select.drawAndUpdate();
      if (ImGui::Button("Load / Reload Videos")) loadVideos();
      ImGui::TextWrapped("%s", video_info.c_str());
      bool dirty = false;
      int offset = calib.sync_offset_frames;
      if (ImGui::InputInt("Sync offset (R vs L)", &offset)) {
        calib.sync_offset_frames = math::clamp(offset, -120, 120);
        seekAndReadFrame(curr_frame);
        dirty = true;
      }
      if (ImGui::Checkbox("Skip first frame (K2 Pro AE quirk)", &calib.skip_first_frame)) {
        seekAndReadFrame(curr_frame);
        dirty = true;
      }
      dirty |= ImGui::Checkbox("Swap eyes", &calib.swap_eyes);
      if (dirty) markPreviewDirty();
    }

    if (ImGui::CollapsingHeader("Lens: Left Eye")) drawEyePanel("Left", calib.left);
    if (ImGui::CollapsingHeader("Lens: Right Eye")) drawEyePanel("Right", calib.right);

    if (ImGui::CollapsingHeader("Rig Alignment", ImGuiTreeNodeFlags_DefaultOpen)) {
      bool dirty = false;
      dirty |= sliderAngle("Global roll (horizon)", &calib.global_roll, -5.0f, 5.0f);
      dirty |= sliderAngle("Convergence", &calib.convergence, -2.0f, 2.0f);
      if (ImGui::Button("Reset rotations")) {
        calib.left.yaw = calib.left.pitch = calib.left.roll = 0;
        calib.right.yaw = calib.right.pitch = calib.right.roll = 0;
        calib.global_roll = calib.convergence = 0;
        dirty = true;
      }
      if (dirty) markPreviewDirty();
    }

    if (ImGui::CollapsingHeader("Auto-Refine", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextWrapped(
          "Solves right-eye pitch & roll by minimizing vertical disparity of matched "
          "features. Do coarse manual alignment first.");
      if (ImGui::Button("Run Auto-Refine") && !command_runner.isRunning()) runAutoRefine();
      ImGui::TextWrapped("%s", getRefineStatus().c_str());
    }

    if (ImGui::CollapsingHeader("Color Match")) {
      ImGui::Text(
          "Gains (BGR): %.3f %.3f %.3f", calib.gain_bgr[0], calib.gain_bgr[1],
          calib.gain_bgr[2]);
      if (ImGui::Button("Match left eye to right")) runColorMatch();
      ImGui::SameLine();
      if (ImGui::Button("Reset gains")) {
        calib.gain_bgr[0] = calib.gain_bgr[1] = calib.gain_bgr[2] = 1.0;
        markPreviewDirty();
      }
    }

    if (ImGui::CollapsingHeader("Preview", ImGuiTreeNodeFlags_DefaultOpen)) {
      bool dirty = false;
      dirty |= ImGui::RadioButton("Anaglyph (red=L cyan=R)", &preview_mode, kAnaglyph);
      dirty |= ImGui::RadioButton("Side by side", &preview_mode, kSBS);
      dirty |= ImGui::RadioButton("50/50 blend", &preview_mode, kBlend);
      dirty |= ImGui::RadioButton("Difference x4", &preview_mode, kDifference);
      dirty |= ImGui::RadioButton("Left only", &preview_mode, kLeftOnly);
      ImGui::SameLine();
      dirty |= ImGui::RadioButton("Right only", &preview_mode, kRightOnly);
      dirty |= ImGui::Checkbox("Latitude gridlines", &show_grid);
      const char* sizes[] = {"512", "768", "1024"};
      static int size_choice = 1;
      if (ImGui::Combo("Preview res", &size_choice, sizes, 3)) {
        preview_size = size_choice == 0 ? 512 : size_choice == 1 ? 768 : 1024;
        dirty = true;
      }
      if (dirty) markPreviewDirty();
    }

    ImGui::EndChild();
  }

  void drawTimeline()
  {
    if (!have_video) return;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 200);
    int f = curr_frame;
    if (ImGui::SliderInt("##frame", &f, 0, std::max(0, num_frames - 1))) {
      curr_frame = f;
      seekAndReadFrame(curr_frame);
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("<")) {
      curr_frame = std::max(0, curr_frame - 1);
      seekAndReadFrame(curr_frame);
    }
    ImGui::SameLine();
    if (ImGui::Button(">")) {
      curr_frame = std::min(num_frames - 1, curr_frame + 1);
      seekAndReadFrame(curr_frame);
    }
    ImGui::SameLine();
    ImGui::Text("Frame %d / %d", curr_frame, num_frames - 1);
  }

  void drawJobStatus()
  {
    if (!command_runner.isRunning()) return;
    const float frac = float(job_progress_frame) / float(std::max(1, int(job_progress_total)));
    ImGui::ProgressBar(frac, ImVec2(ImGui::GetContentRegionAvail().x - 120, 0));
    ImGui::SameLine();
    if (ImGui::Button("Cancel Job")) command_runner.kill();
  }

  void drawModalPopups()
  {
    if (open_popup_export) ImGui::OpenPopup("Export Stitched Video");

    if (ImGui::BeginPopupModal(
            "Export Stitched Video", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      const char* rez[] = {"2048 / eye (4096x2048)", "2880 / eye (5760x2880)",
                           "3840 / eye (7680x3840)"};
      ImGui::Combo("Resolution", &export_eqr_choice, rez, 3);
      const char* codecs[] = {"ProRes 422 HQ (.mov)", "H.265 10-bit (.mov/.mp4)",
                              "PNG sequence (folder)"};
      ImGui::Combo("Codec", &export_codec_choice, codecs, 3);
      ImGui::InputText("Output path", export_path_buf, IM_ARRAYSIZE(export_path_buf));
      ImGui::SameLine();
      if (ImGui::Button("...")) {
        const char* cpath = tinyfd_saveFileDialog(
            "Output video path", "k2_sbs_vr180.mov", 0, nullptr, nullptr);
        if (cpath != nullptr) string::copyBuffer(export_path_buf, cpath, 1024);
      }
      ImGui::Dummy(ImVec2(0, 8));
      const bool ready = have_video && std::string(export_path_buf).size() > 0 &&
                         !command_runner.isRunning();
      if (ImGui::Button("Export", ImVec2(100, 0)) && ready) {
        runExport();
        open_popup_export = false;
      }
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(100, 0))) {
        open_popup_export = false;
      }
      if (!open_popup_export) ImGui::CloseCurrentPopup();
      ImGui::EndPopup();
    }
  }

  void drawFrame() override
  {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin(
        "K2Stitch", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

    drawMainMenu();
    drawToolbar();
    ImGui::SameLine();

    ImGui::BeginChild("##PreviewArea", ImVec2(0, 0), false);
    drawTimeline();
    drawJobStatus();
    if (preview_ready) {
      std::lock_guard<std::mutex> lock(preview_mutex);
      preview_image.setImage(preview_pending);
      preview_ready = false;
    }
    if (!preview_image.empty()) {
      preview_image.scale_to_fit = true;
      preview_image.makeGlTexture();
      preview_image.drawInImGui();
    } else {
      ImGui::Dummy(ImVec2(0, 40));
      ImGui::TextWrapped(
          "Load the left and right eye videos to begin (File menu or Ingest panel). "
          "Then align with the Lens / Rig panels using the anaglyph view: the goal is "
          "zero vertical separation between red and cyan across the gridlines.");
    }
    ImGui::EndChild();

    drawModalPopups();
    ImGui::End();

    finishDrawingImguiAndGl();
  }
};

}  // namespace p11

namespace {

int runBatchMode()
{
  using namespace p11;
  XCHECK(!FLAGS_left.empty() && !FLAGS_right.empty())
      << "--left and --right are required in batch mode";
  XCHECK(!FLAGS_out.empty()) << "--out is required in batch mode";

  k2stitch::RigCalibration calib;
  if (!FLAGS_calib.empty()) {
    XCHECK(k2stitch::loadCalibration(FLAGS_calib, calib))
        << "Failed to load calibration: " << FLAGS_calib;
  } else {
    XPLINFO << "No --calib given; using default (uncalibrated) rig parameters.";
  }

  if (FLAGS_still >= 0) {
    const cv::Mat sbs = k2stitch::stitchStillFromVideos(
        calib, FLAGS_left, FLAGS_right, FLAGS_still, FLAGS_eqr_size);
    XCHECK(!sbs.empty()) << "Failed to stitch still frame " << FLAGS_still;
    cv::imwrite(FLAGS_out, sbs);
    XPLINFO << "Wrote " << FLAGS_out << " (" << sbs.cols << "x" << sbs.rows << ")";
    XPLINFO << "Test in VVE: File -> Import VR180, Render LDI3 (baseline 6.4 cm).";
    return 0;
  }

  k2stitch::ExportConfig cfg;
  cfg.left_path = FLAGS_left;
  cfg.right_path = FLAGS_right;
  cfg.out_path = FLAGS_out;
  cfg.eqr_size = FLAGS_eqr_size;
  cfg.codec = FLAGS_codec;
  cfg.first_frame = FLAGS_first_frame;
  cfg.last_frame = FLAGS_last_frame;
  cfg.fps_override = FLAGS_fps;
  if (!FLAGS_ffmpeg.empty()) {
    cfg.ffmpeg = FLAGS_ffmpeg;
  } else {
#ifdef __APPLE__
    cfg.ffmpeg = "/usr/local/bin/ffmpeg";
#else
    cfg.ffmpeg = "ffmpeg";
#endif
  }
  cfg.cancel = std::make_shared<std::atomic<bool>>(false);
  cfg.progress = [](int frame, int total) {
    if (frame % 30 == 0) XPLINFO << "phase=Stitching frame=" << frame << "/" << total;
  };
  std::string err;
  if (!k2stitch::exportStitched(calib, cfg, err)) {
    XPLINFO << "EXPORT FAILED: " << err;
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  srand(123);  // for imgui file-select hash ids

  if (FLAGS_batch || FLAGS_still >= 0) {
    return runBatchMode();
  }

  p11::K2StitchApp app;
  app.init("K2Stitch by Lifecast (community) - VR180 Align & Stitch", 1720, 1000);
  app.setCharcoalStyle();
  app.initApp();
  app.guiDrawLoop();
  app.shutting_down = true;
  app.cleanup();
  return 0;
}
