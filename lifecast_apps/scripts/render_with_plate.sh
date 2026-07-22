#!/bin/bash
# Render a locked-off VR180 shot to LDI3 using a static-camera background plate,
# then encode an H.264 master.
#
# Phases run separately because the plate must be built from the f-theta frames
# that the depth phase produces, before the inpaint phase consumes it.
#
#   SRC=/workspace/nestt1_1.mov OUT=/workspace/nestt_plate FIRST=0 LAST=767 \
#   bash render_with_plate.sh
set -e

SRC=${SRC:-/workspace/nestt1_1.mov}
OUT=${OUT:-/workspace/nestt_plate}
FIRST=${FIRST:-0}
LAST=${LAST:-767}

# TILE is the OUTPUT resolution per grid tile; the written frame is 3 x TILE
# (1920 -> 5760x5760). WORK is the internal working f-theta resolution: making
# it larger than TILE is supersampling, and 2x is what the VVE GUI does.
TILE=${TILE:-1920}
WORK=${WORK:-$((TILE * 2))}

FPS=${FPS:-30000/1001}
REPO=${REPO:-/workspace/my-4d-video-project/lifecast_apps}
CLI=$REPO/bazel-bin/source/vve_cli
SCRIPTS=$REPO/scripts
PLATE=${PLATE:-/workspace/plate.png}
DPLATE=${DPLATE:-/workspace/plate_depth.png}
OUTFILE=${OUTFILE:-/workspace/nestt1_1_ldi3_h264.mp4}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib

# DA3 model. If it has been fetched into ml_models/ the runfiles default finds
# it and no flag is needed, but existing pods keep it at /workspace, so honour
# that when present rather than silently falling back and failing.
DA3=${DA3:-}
if [ -z "$DA3" ] && [ -f /workspace/da3_stereo.pt ]; then DA3=/workspace/da3_stereo.pt; fi
DA3_FLAG=""
[ -n "$DA3" ] && DA3_FLAG="--da3_model_path $DA3"

# inpaint_method/seg_method/inpaint_dilate_radius/rectified_size_for_depth are
# left at their defaults, which now match the GUI exactly.
COMMON="--src_vr180 $SRC --dest_dir $OUT --depth_method da3_fused $DA3_FLAG \
 --ftheta_size $WORK --inflated_ftheta_size $TILE \
 --output_encoding split12 --first_frame $FIRST --last_frame $LAST"

rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== [1/5] depth phase (work ${WORK}, tile ${TILE} -> $((TILE * 3))x$((TILE * 3))) ==="
T0=$(date +%s)
$CLI $COMMON --phase depth

echo "=== [2/5] stabilize phase ==="
$CLI $COMMON --phase stabilize

echo "=== [3/5] background plate (temporal median) ==="
# build_plates.py, not make_plate.py: it asserts the imwrite return value and
# re-reads each file. make_plate.py ignored the imwrite result, so it could exit
# 0 having written nothing and the render would proceed with no plate at all.
python3 "$SCRIPTS/build_plates.py" "$OUT" "$PLATE" "$DPLATE"

echo "=== [4/5] inpaint + assemble, filling holes from plate ==="
$CLI $COMMON --phase inpaint --plate_path "$PLATE" --depth_plate_path "$DPLATE"
T1=$(date +%s); echo "RENDER_SEC=$((T1-T0))"

N=$(ls "$OUT"/ldi3_*.png | wc -l)
MEAN=$(python3 -c "import cv2,glob;f=sorted(glob.glob('$OUT/ldi3_*.png'));print(round(cv2.imread(f[len(f)//2],-1).mean(),1))")
echo "frames=$N mid_frame_mean=$MEAN"
if [ "${MEAN%%.*}" -lt 1 ]; then echo "ABORT: frames are black"; exit 1; fi

echo "=== [5/5] H.264 encode ==="
ffmpeg -y -framerate $FPS -i "$OUT/ldi3_%06d.png" -c:v libx264 -preset medium \
  -crf 12 -pix_fmt yuv444p -movflags +faststart "$OUTFILE"
T2=$(date +%s); echo "ENCODE_SEC=$((T2-T1))"
cp "$OUT/jg4d_sidecar.json" "${OUTFILE%.mp4}_jg4d_sidecar.json" 2>/dev/null || true
ls -lh "$OUTFILE"
touch /workspace/PLATE_RENDER_DONE
echo ALL_DONE
