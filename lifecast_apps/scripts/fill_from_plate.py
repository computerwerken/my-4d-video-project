#!/usr/bin/env python3
"""Fill masked (disoccluded) regions of a frame by copying from the background
plate, with optional edge feathering so the seam is invisible.

The mask is white (255) where the hole is. Works for 8-bit color frames and
16-bit depth frames alike (frame and plate must share dtype and size).

Usage:
  python3 fill_from_plate.py --image frame.png --mask holes.png \
      --plate plate.png --out filled.png [--feather 3]
Batch mode (fill every frame of a shot with one plate):
  python3 fill_from_plate.py --image_dir frames/ --mask_dir masks/ \
      --plate plate.png --out_dir filled/ [--feather 3]
"""
import argparse, glob, os, sys
import cv2
import numpy as np


def fill_one(image_path, mask_path, plate, feather, out_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        sys.exit(f"Cannot read {image_path} or {mask_path}")
    if img.shape[:2] != plate.shape[:2] or img.dtype != plate.dtype:
        sys.exit(f"Frame/plate mismatch: {img.shape} {img.dtype} vs {plate.shape} {plate.dtype}")

    alpha = (msk.astype(np.float32) / 255.0)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
        alpha = np.maximum(alpha, (msk > 127).astype(np.float32))  # holes stay fully filled
    if img.ndim == 3:
        alpha = alpha[..., None]

    out = img.astype(np.float32) * (1.0 - alpha) + plate.astype(np.float32) * alpha
    cv2.imwrite(out_path, out.astype(img.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", required=True)
    ap.add_argument("--feather", type=float, default=3.0)
    ap.add_argument("--image"); ap.add_argument("--mask"); ap.add_argument("--out")
    ap.add_argument("--image_dir"); ap.add_argument("--mask_dir"); ap.add_argument("--out_dir")
    args = ap.parse_args()

    plate = cv2.imread(args.plate, cv2.IMREAD_UNCHANGED)
    if plate is None:
        sys.exit(f"Cannot read plate {args.plate}")

    if args.image:
        fill_one(args.image, args.mask, plate, args.feather, args.out)
        print(f"Wrote {args.out}")
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        images = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
        for ip in images:
            name = os.path.basename(ip)
            mp = os.path.join(args.mask_dir, name)
            if not os.path.exists(mp):
                print(f"  no mask for {name}, copying through"); mp = None
            op = os.path.join(args.out_dir, name)
            if mp is None:
                cv2.imwrite(op, cv2.imread(ip, cv2.IMREAD_UNCHANGED))
            else:
                fill_one(ip, mp, plate, args.feather, op)
        print(f"Filled {len(images)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
