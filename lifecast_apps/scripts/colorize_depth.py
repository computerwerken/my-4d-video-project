#!/usr/bin/env python3
"""Colorize 16-bit depth PNGs so a human can judge them (see walkthrough Stage 1).

Usage: python3 colorize_depth.py depth_000123.png [more.png ...]
Writes <name>_color.png next to each input.
"""
import sys
import cv2

for path in sys.argv[1:]:
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        print(f"skip (unreadable): {path}"); continue
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    m = d.max()
    if m == 0:
        print(f"skip (all zero): {path}"); continue
    c = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255.0 / m), cv2.COLORMAP_TURBO)
    out = path.rsplit(".", 1)[0] + "_color.png"
    cv2.imwrite(out, c)
    print(f"wrote {out}")
