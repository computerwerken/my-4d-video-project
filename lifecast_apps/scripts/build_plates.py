#!/usr/bin/env python3
"""Build colour + depth background plates for a locked-off shot, with explicit
diagnostics and hard asserts on every write. Replaces make_plate.py in the plate
workflow after make_plate was observed to exit 0 without producing a readable
file at full resolution.

  python3 build_plates.py FRAMES_DIR
Writes /workspace/plate.png (colour) and /workspace/plate_depth.png (depth).
"""
import glob
import os
import sys

import cv2
import numpy as np

frames_dir = sys.argv[1] if len(sys.argv) > 1 else '/workspace/nestt_plate'
MAX = 121


def build(pattern, out, unchanged):
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    print(f'[{pattern}] found {len(files)} files', flush=True)
    if not files:
        raise SystemExit(f'no files for {pattern}')
    if len(files) > MAX:
        idx = np.linspace(0, len(files) - 1, MAX).round().astype(int)
        files = [files[i] for i in sorted(set(idx))]
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    first = cv2.imread(files[0], flag)
    if first is None:
        raise SystemExit(f'cannot read {files[0]}')
    print(f'  frame shape {first.shape} dtype {first.dtype}; median over {len(files)}', flush=True)
    H = first.shape[0]
    strip = 256
    plate = np.zeros_like(first)
    for y0 in range(0, H, strip):
        y1 = min(y0 + strip, H)
        buf = np.empty((len(files), y1 - y0) + first.shape[1:], dtype=first.dtype)
        for i, f in enumerate(files):
            im = cv2.imread(f, flag)
            if im is None or im.shape != first.shape:
                raise SystemExit(f'bad frame {f}')
            buf[i] = im[y0:y1]
        plate[y0:y1] = np.median(buf, axis=0).astype(first.dtype)
    plate = np.ascontiguousarray(plate)
    ok = cv2.imwrite(out, plate)
    print(f'  imwrite({out}) -> {ok}', flush=True)
    if not ok:
        raise SystemExit(f'imwrite returned False for {out}')
    check = cv2.imread(out, cv2.IMREAD_UNCHANGED)
    if check is None:
        raise SystemExit(f'wrote {out} but it is not readable')
    print(f'  VERIFIED {out} {check.shape} {check.dtype} mean {round(float(check.mean()),1)}', flush=True)


build('R_ftheta_[0-9]*.png', '/workspace/plate.png', unchanged=False)
build('filtered_R_depth_[0-9]*.png', '/workspace/plate_depth.png', unchanged=True)
print('BUILD_PLATES_DONE', flush=True)
