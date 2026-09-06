#!/usr/bin/env python3
"""Fast-FoundationStereo disparity daemon for `vve_cli --stereo_backend external`.

Protocol (everything lives in <dest_dir>/xstereo, passed as --dir):
  vve_cli writes   xs_L_NNNNNN.png + xs_R_NNNNNN.png   8-bit BGR rectified pair
  this script writes xs_D_NNNNNN.png                   uint16 = disparity_px * --scale,
                                                        referenced to the RIGHT image
  vve_cli deletes xs_D after reading; we delete xs_L / xs_R after processing.
All writes go to a .tmp name and are renamed, so nobody ever reads a partial PNG.
Stop with a file named STOP in --dir, or Ctrl-C.

Right-referenced disparity via the flip trick: FFS predicts disparity for its *left*
input, so we feed (hflip(R), hflip(L)) and hflip the result back. Same magnitude,
same sign, referenced to R -- exactly what VVE's RAFT path produces.

Usage (pod):
  source /workspace/ffs_venv/bin/activate
  python ffs_daemon.py --dir /workspace/out_x/xstereo
"""
import argparse
import glob
import logging
import os
import sys
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='<dest_dir>/xstereo of the vve_cli run')
    ap.add_argument('--ffs_root', default='/workspace/ffs', help='Fast-FoundationStereo checkout')
    ap.add_argument('--model', default='/workspace/ffs/weights/23-36-37/model_best_bp2_serialize.pth')
    ap.add_argument('--valid_iters', type=int, default=8)
    ap.add_argument('--max_disp', type=int, default=416)
    ap.add_argument('--hiera', type=int, default=0, help='hierarchical inference (for >1K inputs)')
    ap.add_argument('--low_memory', type=int, default=0, help='chunked cost-volume lookup; auto-enabled on cuDNN failure (needed >= ~2K)')
    ap.add_argument('--scale', type=float, default=32.0, help='uint16 = disparity_px * scale (must match vve_cli --external_disparity_scale)')
    ap.add_argument('--poll', type=float, default=0.02, help='idle sleep seconds')
    ap.add_argument('--keep_inputs', type=int, default=0, help='1 = leave xs_L/xs_R on disk (debug)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

    sys.path.insert(0, args.ffs_root)
    import torch  # noqa: E402
    from core.utils.utils import InputPadder  # noqa: E402
    from Utils import AMP_DTYPE  # noqa: E402

    torch.autograd.set_grad_enabled(False)
    model = torch.load(args.model, map_location='cpu', weights_only=False)
    model.args.valid_iters = args.valid_iters
    model.args.max_disp = args.max_disp
    model.cuda().eval()
    logging.info(f'model loaded: {args.model}  iters={args.valid_iters} max_disp={args.max_disp} hiera={args.hiera}')

    def infer(left_rgb, right_rgb, low_memory):
        H, W = left_rgb.shape[:2]
        a = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        b = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            if args.hiera:
                disp = model.run_hierachical(a, b, iters=args.valid_iters, test_mode=True,
                                             low_memory=low_memory, small_ratio=0.5)
            else:
                disp = model.forward(a, b, iters=args.valid_iters, test_mode=True,
                                     low_memory=low_memory, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        return disp.data.cpu().numpy().reshape(H, W)

    def infer_robust(left_rgb, right_rgb):
        # cuDNN grid_sample rejects the full-batch cost-volume lookup above ~1.6K inputs
        # (CUDNN_STATUS_NOT_SUPPORTED); low_memory chunks it. Try fast path first.
        try:
            return infer(left_rgb, right_rgb, bool(args.low_memory))
        except RuntimeError as e:
            if args.low_memory:
                raise
            logging.warning(f'fast path failed ({str(e).splitlines()[0][:80]}); switching to --low_memory 1 for the rest of the run')
            args.low_memory = 1
            torch.cuda.empty_cache()
            return infer(left_rgb, right_rgb, True)

    os.makedirs(args.dir, exist_ok=True)
    stop_path = os.path.join(args.dir, 'STOP')
    logging.info(f'watching {args.dir}  (touch {stop_path} to exit)')
    n = 0
    while not os.path.exists(stop_path):
        did = False
        for Lp in sorted(glob.glob(os.path.join(args.dir, 'xs_L_*.png'))):
            fnum = os.path.basename(Lp)[5:-4]
            Rp = os.path.join(args.dir, f'xs_R_{fnum}.png')
            Dp = os.path.join(args.dir, f'xs_D_{fnum}.png')
            if not os.path.exists(Rp) or os.path.exists(Dp):
                continue
            L = cv2.imread(Lp, cv2.IMREAD_COLOR)
            R = cv2.imread(Rp, cv2.IMREAD_COLOR)
            if L is None or R is None or L.shape != R.shape:
                logging.warning(f'{fnum}: unreadable/mismatched pair, retrying')
                time.sleep(0.1)
                continue
            t0 = time.time()
            try:
                # [:, ::-1, ::-1] = horizontal flip + BGR->RGB in one go.
                d = infer_robust(np.ascontiguousarray(R[:, ::-1, ::-1]), np.ascontiguousarray(L[:, ::-1, ::-1]))
            except Exception:
                # Leave a note vve_cli can fail fast on instead of waiting for its timeout.
                import traceback
                msg = traceback.format_exc()
                logging.error(f'{fnum}: inference failed\n{msg}')
                with open(os.path.join(args.dir, f'xs_E_{fnum}.txt'), 'w') as f:
                    f.write(msg)
                continue
            d = d[:, ::-1]
            d16 = np.clip(d * args.scale, 0, 65535).astype(np.uint16)
            tmp = Dp + '.tmp.png'
            cv2.imwrite(tmp, d16)
            os.replace(tmp, Dp)
            if not args.keep_inputs:
                os.remove(Lp)
                os.remove(Rp)
            n += 1
            did = True
            logging.info(f'{fnum}: {d.shape[1]}x{d.shape[0]}  disp max {d.max():.1f} px  {time.time() - t0:.2f}s  (#{n})')
        if not did:
            time.sleep(args.poll)
    logging.info(f'STOP seen, exiting after {n} frames')


if __name__ == '__main__':
    main()
