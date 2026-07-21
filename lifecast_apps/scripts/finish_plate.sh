#!/bin/bash
# Resume the plate render: build both plates from the already-computed depth/
# stabilize output, run inpaint with them, encode H.264. Depth+stabilize are
# already done in /workspace/nestt_plate.
set -e
OUT=/workspace/nestt_plate
S=/workspace/my-4d-video-project/lifecast_apps/scripts
CLI=/workspace/my-4d-video-project/lifecast_apps/bazel-bin/source/vve_cli
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib
COMMON="--src_vr180 /workspace/nestt1_1.mov --dest_dir $OUT --depth_method da3_fused \
 --da3_model_path /workspace/da3_stereo.pt --inpaint_method ceres --seg_method heuristic \
 --ftheta_size 1920 --inflated_ftheta_size 1920 --output_encoding split12 \
 --first_frame 0 --last_frame 767"

echo "=== colour plate ==="
python3 $S/make_plate.py --frames_dir $OUT --pattern 'R_ftheta_[0-9]*.png' \
  --out /workspace/plate.png --max_samples 121 --strip_height 256
echo "=== depth plate ==="
python3 $S/make_plate.py --frames_dir $OUT --pattern 'filtered_R_depth_[0-9]*.png' \
  --out /workspace/plate_depth.png --max_samples 121 --strip_height 256
python3 -c "import cv2;c=cv2.imread('/workspace/plate.png');d=cv2.imread('/workspace/plate_depth.png',-1);print('colour',c.shape,c.dtype,'depth',d.shape,d.dtype)"

echo "=== inpaint with plates ==="
LIFECAST_PLATE_PATH=/workspace/plate.png LIFECAST_DEPTH_PLATE_PATH=/workspace/plate_depth.png \
  $CLI $COMMON --phase inpaint

N=$(ls $OUT/ldi3_*.png | wc -l)
MEAN=$(python3 -c "import cv2,glob;f=sorted(glob.glob('$OUT/ldi3_*.png'));print(round(cv2.imread(f[len(f)//2],-1).mean(),1))")
echo "frames=$N mid_mean=$MEAN"
[ "${MEAN%%.*}" -lt 1 ] && { echo ABORT_BLACK; exit 1; }

echo "=== H.264 encode ==="
ffmpeg -y -framerate 30000/1001 -i $OUT/ldi3_%06d.png -c:v libx264 -preset medium \
  -crf 12 -pix_fmt yuv444p -movflags +faststart /workspace/nestt1_1_ldi3_h264.mp4
cp $OUT/jg4d_sidecar.json /workspace/nestt1_1_jg4d_sidecar.json 2>/dev/null || true
ls -lh /workspace/nestt1_1_ldi3_h264.mp4
touch /workspace/PLATE_RENDER_DONE
echo ALL_DONE
