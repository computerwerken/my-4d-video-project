#!/bin/bash
# Robust finisher: build colour+depth plates (build_plates.py, self-verifying),
# run inpaint with both plates, sanity-check, encode H.264. Aborts loudly on any
# failure instead of producing garbage.
set -uo pipefail
OUT=/workspace/nestt_plate
S=/workspace/my-4d-video-project/lifecast_apps/scripts
CLI=/workspace/my-4d-video-project/lifecast_apps/bazel-bin/source/vve_cli
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/torch/lib
die(){ echo "FINISH_FAIL: $*"; exit 1; }

echo "=== build plates ==="
python3 "$S/build_plates.py" "$OUT" || die "build_plates exit $?"

echo "=== inpaint with plates ==="
LIFECAST_PLATE_PATH=/workspace/plate.png LIFECAST_DEPTH_PLATE_PATH=/workspace/plate_depth.png \
  $CLI --src_vr180 /workspace/nestt1_1.mov --dest_dir "$OUT" --depth_method da3_fused \
  --da3_model_path /workspace/da3_stereo.pt --inpaint_method ceres --seg_method heuristic \
  --ftheta_size 1920 --inflated_ftheta_size 1920 --output_encoding split12 \
  --first_frame 0 --last_frame 767 --phase inpaint || die "inpaint exit $?"

N=$(ls "$OUT"/ldi3_*.png 2>/dev/null | wc -l)
[ "$N" -eq 768 ] || die "expected 768 ldi3 frames, got $N"
MEAN=$(python3 -c "import cv2,glob;f=sorted(glob.glob('$OUT/ldi3_*.png'));print(int(cv2.imread(f[len(f)//2],-1).mean()))")
echo "frames=$N mid_mean=$MEAN"
[ "$MEAN" -ge 1 ] || die "frames are black (mean $MEAN)"

echo "=== H.264 encode ==="
ffmpeg -y -framerate 30000/1001 -i "$OUT/ldi3_%06d.png" -c:v libx264 -preset medium \
  -crf 12 -pix_fmt yuv444p -movflags +faststart /workspace/nestt1_1_ldi3_h264.mp4 \
  >/workspace/encode.log 2>&1 || die "ffmpeg exit $?"
cp "$OUT/jg4d_sidecar.json" /workspace/nestt1_1_jg4d_sidecar.json 2>/dev/null || true
ls -lh /workspace/nestt1_1_ldi3_h264.mp4
touch /workspace/PLATE_RENDER_DONE
echo ALL_DONE
