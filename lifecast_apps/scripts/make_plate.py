#!/usr/bin/env python3
"""Build a background plate from a stationary-camera shot by temporal median.

Wherever a foreground object MOVED at some point during the shot, the true
background survives the per-pixel median. Objects that never move remain in the
plate (fill those once with fill_from_plate.py + LaMa and your own eyeballs).

Works on 8-bit color frames AND 16-bit depth PNGs (use it twice: color plate
and depth plate). Memory-safe for 8K frames via horizontal-strip processing and
uniform frame sampling (a median over ~151 well-spread frames is statistically
the same as over all 18,000).

Usage:
  python3 make_plate.py --frames_dir shot01_frames --pattern "*.png" --out plate.png
  python3 make_plate.py --video shot01.mov --out plate.png          # extracts frames itself
Options:
  --max_samples 151    frames used for the median (odd number recommended)
  --strip_height 256   rows processed at a time (lower = less RAM)
"""
import argparse, glob, os, subprocess, sys, tempfile
import cv2
import numpy as np


def list_frames(frames_dir, pattern):
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not files:
        sys.exit(f"No frames matching {pattern} in {frames_dir}")
    return files


def sample(files, max_samples):
    if len(files) <= max_samples:
        return files
    idx = np.linspace(0, len(files) - 1, max_samples).round().astype(int)
    return [files[i] for i in sorted(set(idx))]


def extract_video(video, tmpdir):
    out = os.path.join(tmpdir, "f_%06d.png")
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, out], check=True)
    return list_frames(tmpdir, "f_*.png")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames_dir")
    src.add_argument("--video")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=151)
    ap.add_argument("--strip_height", type=int, default=256)
    args = ap.parse_args()

    tmpdir = None
    if args.video:
        tmpdir = tempfile.mkdtemp(prefix="plate_")
        files = extract_video(args.video, tmpdir)
    else:
        files = list_frames(args.frames_dir, args.pattern)
    files = sample(files, args.max_samples)
    print(f"Median over {len(files)} sampled frames")

    first = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        sys.exit(f"Cannot read {files[0]}")
    H = first.shape[0]
    plate = np.zeros_like(first)

    for y0 in range(0, H, args.strip_height):
        y1 = min(y0 + args.strip_height, H)
        strips = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape != first.shape:
                sys.exit(f"Bad/mismatched frame: {f}")
            strips.append(img[y0:y1])
        plate[y0:y1] = np.median(np.stack(strips, axis=0), axis=0).astype(first.dtype)
        print(f"  rows {y0}-{y1} done")

    cv2.imwrite(args.out, plate)
    print(f"Wrote {args.out}  ({plate.shape}, {plate.dtype})")
    print("Now LOOK at it: moving people should be erased; anything that never "
          "moved is still there and needs a one-time manual/LaMa fill.")


if __name__ == "__main__":
    main()
