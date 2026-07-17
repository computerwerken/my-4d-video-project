#!/usr/bin/env python3
"""List the shot boundaries (cuts) in an edited video using ffmpeg's scene
detection. Prints frame numbers + timestamps; use them to split the movie into
stationary shots for per-shot background plates.

Usage:
  python3 detect_shots.py --video my_edit.mov [--threshold 0.3] [--fps 30]
Hand-verify the list: threshold 0.3 catches hard cuts; dissolves may need 0.2.
"""
import argparse, re, subprocess, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--fps", type=float, default=30.0, help="for frame-number conversion")
    args = ap.parse_args()

    cmd = ["ffmpeg", "-i", args.video, "-vf",
           f"select='gt(scene,{args.threshold})',showinfo", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", r.stderr)]

    print(f"{len(times)} cut(s) detected (threshold {args.threshold})")
    print("shot  start_time  start_frame")
    starts = [0.0] + times
    for i, t in enumerate(starts):
        print(f"{i+1:4d}  {t:10.3f}  {int(round(t * args.fps)):11d}")
    if not times:
        print("(single continuous shot)")


if __name__ == "__main__":
    main()
