#!/usr/bin/env python3
"""Prepare 3x Canon R5C dual-fisheye recordings for lifecast_splat video mode.

Slices each side-by-side dual-fisheye recording into per-eye videos with
ffmpeg (no EOS VR Utility - the pipeline wants RAW fisheye + an equiangular
camera model), then writes the vid_dir layout that lifecast_splat expects
(see lifecast_splat_lib.cc:1059): dataset.json, time_offsets.json (zeros,
hardware-synced rig), and cam{0,1,2}{L,R}.mp4.

The emitted extrinsics are a nominal SEED (cameras in a row along +x, eyes
60 mm apart) and MUST be refined per shot with rig_calibrate.sh - the rig
shifted between shots. Intrinsics are seeds too; SfM refines them with
--intrinsic_prior. world_from_cam is written flat COLUMN-major, matching
Eigen::Map in multicamera_dataset.cc:28.

Usage:
  python3 rig_prepare.py camA.mov camB.mov camC.mov \
      --out /workspace/rigtest --start 00:00:10 --dur 10
"""

import argparse
import json
import math
import os
import subprocess
import sys

CAM_NAMES = ["cam0L", "cam0R", "cam1L", "cam1R", "cam2L", "cam2R"]


def ffprobe_dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), s["r_frame_rate"]


def slice_eye(src, dst, side, w, h, args):
    """Crop one eye (left or right half) and transcode a sync'd window."""
    x = 0 if side == "L" else w // 2
    vf = f"crop={w//2}:{h}:{x}:0"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    if args.start:
        cmd += ["-ss", args.start]
    cmd += ["-i", src]
    if args.dur:
        cmd += ["-t", str(args.dur)]
    cmd += ["-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
            "-pix_fmt", "yuv444p", dst]
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not os.path.isfile(dst) or os.path.getsize(dst) < 1000:
        raise SystemExit(f"slice failed or wrote a stub: {dst}")


def seed_pose_world_from_cam(pair_idx, eye, args):
    """Nominal rig pose: pairs along +x, identity rotation. SEED ONLY."""
    x = pair_idx * args.pair_spacing + (-args.baseline / 2 if eye == "L"
                                        else args.baseline / 2)
    m = [[1, 0, 0, x],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    # flatten COLUMN-major (Eigen::Map<Matrix4d> default storage)
    return [m[r][c] for c in range(4) for r in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs=3, help="the 3 dual-fisheye recordings, "
                    "in rig order left-to-right (pair0, pair1, pair2)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="", help="sync'd start timecode, e.g. 00:00:10")
    ap.add_argument("--dur", type=float, default=10.0, help="seconds (0 = all)")
    ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--baseline", type=float, default=0.060,
                    help="intra-pair eye distance, metres")
    ap.add_argument("--pair_spacing", type=float, default=0.20,
                    help="nominal distance between pair centres, metres (SEED)")
    ap.add_argument("--fov_deg", type=float, default=190.0,
                    help="lens FOV; RF5.2mm dual fisheye = 190")
    ap.add_argument("--circle_frac", type=float, default=0.94,
                    help="image-circle diameter as a fraction of frame height")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    dims = [ffprobe_dims(v) for v in args.videos]
    w, h, fps = dims[0]
    for i, d in enumerate(dims):
        if (d[0], d[1]) != (w, h):
            raise SystemExit(f"resolution mismatch: {args.videos[i]} is "
                             f"{d[0]}x{d[1]}, expected {w}x{h}")
    print(f"input: {w}x{h} @ {fps}; per-eye {w//2}x{h}")

    # --- slice all six eyes ---------------------------------------------------
    for pair_idx, src in enumerate(args.videos):
        for eye in ("L", "R"):
            name = f"cam{pair_idx}{eye}"
            print(f"[{name}] <- {os.path.basename(src)} ({eye} half)")
            slice_eye(src, os.path.join(args.out, name + ".mp4"), eye, w, h, args)

    # --- intrinsics seed (equiangular model, multicamera_dataset.cc:47) ------
    eye_w, eye_h = w // 2, h
    circle_radius = args.circle_frac * eye_h / 2.0
    # equiangular: r = radius_at_90 * theta / 90deg; the circle edge sits at
    # fov/2, so radius_at_90 = circle_radius * 90 / (fov/2)
    radius_at_90 = circle_radius * 90.0 / (args.fov_deg / 2.0)

    frames = []
    for pair_idx in range(3):
        for eye in ("L", "R"):
            name = f"cam{pair_idx}{eye}"
            frames.append({
                "cam_model": "equiangular",
                "image_filename": name,     # video mode: <name>.mp4
                "width": eye_w,
                "height": eye_h,
                "radius_at_90": radius_at_90,
                "useable_radius": circle_radius,
                "cx": eye_w / 2.0,          # SEED; refine with --intrinsic_prior
                "cy": eye_h / 2.0,
                "k1": 0.0,
                "tilt": 0.0,
                "world_from_cam": seed_pose_world_from_cam(pair_idx, eye, args),
            })

    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump({"frames_data": frames, "_note":
                   "extrinsics+intrinsics are SEEDS from rig_prepare.py; "
                   "run rig_calibrate.sh per shot before splat training"},
                  f, indent=2)
    with open(os.path.join(args.out, "time_offsets.json"), "w") as f:
        json.dump({n: 0 for n in CAM_NAMES}, f, indent=2)

    # verify everything landed
    for n in CAM_NAMES:
        p = os.path.join(args.out, n + ".mp4")
        if not os.path.isfile(p):
            raise SystemExit(f"missing output: {p}")
    print(f"RIG_PREPARE_DONE {args.out} "
          f"(radius_at_90={radius_at_90:.1f}px, circle={circle_radius:.1f}px)")


if __name__ == "__main__":
    sys.exit(main())
