# Three-R5C rig → 4D Gaussian → LDI3 pipeline

Design for processing the hardwire-synchronized 3x Canon R5C dual-fisheye rig
(6 views: 3 stereo pairs, RF5.2mm, ~60 mm intra-pair baseline) through the
in-repo 4D Gaussian Studio machinery into LDI3. Written against the code as of
this commit; every claim below was checked against the source, with file:line
references so it can be re-verified.

## The key finding: video mode is already a multi-camera rig pipeline

`lifecast_splat` video mode (`--vid_dir`) does NOT assume a single moving
camera. It reads (`lifecast_splat_lib.cc:1059-1095`):

    vid_dir/
      dataset.json         # one entry per PHYSICAL camera (pose + intrinsics)
      time_offsets.json    # per-camera frame offset (hardware sync -> all 0)
      masks/<name>.png     # optional per-camera ignore masks
      <name>.mp4           # one video PER CAMERA

Each frame index is read from every camera's mp4, fisheye views are rectified
internally (`precomputeFisheyeToRectilinearWarp`, `lifecast_splat_lib.cc:1018`),
frame 0 gets `first_frame_warmup_itrs`, later frames fine-tune from the
previous model (`:1170-1174`), and per-frame encoded splats land in
`splat_frames/%06d.png`. This is exactly the rig's shape. No COLMAP anywhere.

## Answers to the specific design questions

**Slice with FFmpeg instead of EOS VR Utility — yes, and it's the better path.**
The pipeline consumes *raw fisheye* with an `equiangular` camera model
(`multicamera_dataset.cc:47-58`); de-warping through EOS VR Utility would
destroy the geometry the calibration expects and resample the image twice.
Crop the SBS recording into left/right halves per camera and keep full
resolution. Bake the LUT in the Premiere export as planned — the splat trainer
fits whatever colours it is given, and graded footage gives it more usable
texture than log.

**COLMAP-with-fisheye pain is avoided entirely.** The in-repo incremental SfM
is templated on camera type with a native fisheye branch
(`incremental_sfm_lib.h:276` `is_fisheye_camera_v`), already exercised by the
4DGStudio GUI (`4dgstudio.cc:1441-1469`, `CAMERA_TYPE_FISHEYE`). The headless
CLI (`incremental_sfm.cc`) was hardcoded to rectilinear with the fisheye
vector literally commented out at line 100 — this commit adds `--fisheye` and
`--dist_a_to_b` flags mirroring the GUI branch.

**Shot-to-shot rig shift → calibrate per shot, it's cheap.** The cameras are
static within a shot, so SfM needs only ONE synchronized 6-frame set per shot
(a few more for robustness). Feed the 6 synced frames to the SfM as a 6-frame
sequence with `time_window_size=0` (all-pairs matching), `--fisheye`,
`--share_intrinsics` (six identical lenses), `--intrinsic_prior`. Solve time is
seconds. Re-run per shot; splat training then uses that shot's `dataset.json`.

**Metric scale for free.** `estimateCameraPosesAndKeypoint3DWithIncrementalSfm`
takes `dist_a_to_b` — the enforced distance between the first two cameras
(previously hardcoded 0 = auto in `incremental_sfm.cc:107`). Order the synced
frames cam0L, cam0R, ... and pass `--dist_a_to_b 0.060`: the reconstruction
comes out in metres, which keeps `inv_depth_coef`/LDI3 depth encoding
consistent with the stereo pipeline.

**DA3 + Ceres integration points.**
- Ceres is already the SfM solver; nothing to do.
- The splat trainer has a depth loss with per-image learnable scale+bias
  (`use_depth_loss`, `lifecast_splat_lib.cc:254-337`, comment says "from
  MiDaS") and `init_with_monodepth` for population init. Substitute DA3:
  our `da3_stereo.pt` gives stereo-consistent depth per pair, strictly better
  than mono MiDaS. Feed each eye's DA3-fused depth as its prior.
- The ceres inpaint + background-plate machinery applies at the LDI3 assembly
  stage (below), unchanged.

## The missing piece: splat → LDI3

4DGStudio exports only the web-player splat format
(`saveEncodedSplatFileWithSizzleZ`, `lifecast_splat_io.h:33`); there is no
LDI3 exporter anywhere in the tree. The bridge (next build step, on the pod):

`splat_to_ldi3` CLI, per frame:
1. Load `splat_frames/%06d.png`.
2. Render RGB + depth from a virtual camera at the rig centre. The gsplat
   rasterizer already outputs depth per pixel (`gsplat_lib.cc:770`
   concatenates depth into the colour channels), but only for rectilinear
   cameras — so render a 5-face cubemap (skip behind) and warp the faces into
   an f-theta image + inverse-depth map, exactly inverse to what
   `precomputeFisheyeToRectilinearWarp` does.
3. Feed `vve_cli --photo_mode --src_ftheta_image --src_ftheta_depth` — the
   existing, just-validated LDI pipeline builds the 3-layer grid with ceres
   inpainting, plates, split12, sidecar. All of this session's fixes apply.

This reuses the entire validated LDI3 stack; the only new code is the cubemap
render + warp (~200 lines against existing helpers).

Phase 2 (optional, later): depth-sliced direct splat→3-layer render for true
volumetric layers instead of inpainted ones.

## Ten-second test plan (when jgdrive footage is exported)

1. `python3 scripts/rig_prepare.py A.mov B.mov C.mov --out /workspace/rigtest
   --start 00:00:10 --dur 10` → 6 sliced mp4s + skeleton `dataset.json`
   (seed intrinsics/extrinsics) + zero `time_offsets.json`.
2. `bash scripts/rig_calibrate.sh /workspace/rigtest` → per-shot SfM refine →
   final `dataset.json` (+ `sfm_pointcloud` for splat init).
3. `lifecast_splat --vid_dir /workspace/rigtest --output_dir /workspace/rigout
   --init_with_monodepth --use_depth_loss` (DA3 priors once wired) →
   `splat_frames/*.png`. ~300 frames; warmup 2000 itrs + per-frame fine-tune.
4. `splat_to_ldi3` (once built) → LDI3 grid frames → H.264 5760x5760 master.

## Known-unknowns to verify on the pod (in order)

- Axis convention of `world_from_cam` (column-major flat 16,
  `multicamera_dataset.cc:27-28`) vs the rig's forward direction — the seed
  extrinsics in `rig_prepare.py` are marked SEED and replaced by SfM anyway.
- Whether video mode actually consumes `use_depth_loss` inputs (the MiDaS
  loader may be static-mode only) — if not, DA3 priors enter via
  `init_with_monodepth` seeding only.
- R5C dual-fisheye circle geometry (centre offsets per eye) — seed values in
  `rig_prepare.py` are tunable flags; `--intrinsic_prior` refinement absorbs
  small errors.
- `incremental_sfm --fisheye` compiles and converges on a 6-view set (the
  template is GUI-proven, the flag is new).
