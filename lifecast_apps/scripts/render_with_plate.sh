#!/bin/bash
# Render a locked-off VR180 shot to LDI3 using a static-camera background plate,
# then encode an H.264 master. Assumes the plate patch is applied to
# ldi_common.cc (scripts/patch_plate_inpaint.py) and vve_cli is built.
#
# Phases are run separately because the plate must be built from the f-theta
# frames that the depth phase produces, before the inpaint phase consumes it.
#
#   SRC=/workspace/nestt1_1.mov OUT=/workspace/nestt_plate FIRST=0 LAST=767 \
#   FTHETA=1920 bash render_with_plate.sh
set -e

SRC=${SRC:-/workspace/nestt1_1.mov}
OUT=${OUT:-/workspace/nestt_plate}
FIRST=${FIRST:-0}
LAST=${LAST:-767}
FTHETA=${FTHETA:-1920}          # LDI3 grid is 3 x FTHETA (1920 -> 5760x5760)
FPS=${FPS:-30000/1001}
CLI=/workspace/my-4d-video-project/lifecast_apps/bazel-bin/source/vve_cli
SCRIPTS=/workspace/my-4d-video-project/lifecast_apps/scripts
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib

COMMON="--src_vr180 $SRC --dest_dir $OUT --depth_method da3_fused \
 --da3_model_path /workspace/da3_stereo.pt --inpaint_method ceres \
 --seg_method heuristic --ftheta_size $FTHETA --inflated_ftheta_size $FTHETA \
 --output_encoding split12 --first_frame $FIRST --last_frame $LAST"

rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== [1/5] depth phase ==="
T0=$(date +%s)
$CLI $COMMON --phase depth

echo "=== [2/5] stabilize phase ==="
$CLI $COMMON --phase stabilize

echo "=== [3/5] background plate (temporal median) ==="
python3 "$SCRIPTS/make_plate.py" --frames_dir "$OUT" --pattern 'R_ftheta_0*.png' \
  --out /workspace/plate.png --max_samples 121 --strip_height 256
python3 -c "import cv2;a=cv2.imread('/workspace/plate.png');print('colour plate',a.shape,'mean',round(a.mean(),1))"
python3 "$SCRIPTS/make_plate.py" --frames_dir "$OUT" --pattern 'filtered_R_depth_*.png' \
  --out /workspace/plate_depth.png --max_samples 121 --strip_height 256
python3 -c "import cv2;a=cv2.imread('/workspace/plate_depth.png',-1);print('depth plate',a.shape,a.dtype)"

echo "=== [4/5] inpaint + assemble, filling holes from plate ==="
LIFECAST_PLATE_PATH=/workspace/plate.png LIFECAST_DEPTH_PLATE_PATH=/workspace/plate_depth.png $CLI $COMMON --phase inpaint
T1=$(date +%s); echo "RENDER_SEC=$((T1-T0))"

N=$(ls "$OUT"/ldi3_*.png | wc -l)
MEAN=$(python3 -c "import cv2,glob;f=sorted(glob.glob('$OUT/ldi3_*.png'));print(round(cv2.imread(f[len(f)//2],-1).mean(),1))")
echo "frames=$N mid_frame_mean=$MEAN"
if [ "${MEAN%%.*}" -lt 1 ]; then echo "ABORT: frames are black"; exit 1; fi

echo "=== [5/5] H.264 encode ==="
ffmpeg -y -framerate $FPS -i "$OUT/ldi3_%06d.png" -c:v libx264 -preset medium \
  -crf 12 -pix_fmt yuv444p -movflags +faststart /workspace/nestt1_1_ldi3_h264.mp4
T2=$(date +%s); echo "ENCODE_SEC=$((T2-T1))"
cp "$OUT/jg4d_sidecar.json" /workspace/nestt1_1_jg4d_sidecar.json 2>/dev/null || true
ls -lh /workspace/nestt1_1_ldi3_h264.mp4
touch /workspace/PLATE_RENDER_DONE
echo ALL_DONE
