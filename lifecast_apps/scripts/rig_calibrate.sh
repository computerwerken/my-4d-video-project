#!/usr/bin/env bash
# Per-shot extrinsics refinement for the 3x R5C rig (see RIG_PIPELINE.md).
#
# The rig shifted between shots, so run this once per shot on the vid_dir that
# rig_prepare.py produced. It extracts ONE hardware-synchronized frame from
# each of the 6 per-eye videos, assembles them into a 6-frame sequence, runs
# the in-repo fisheye SfM (incremental_sfm --fisheye) with the 60 mm intra-pair
# baseline as metric scale, and merges the refined poses+intrinsics back into
# dataset.json (backing up the seed as dataset.seed.json).
#
# Usage:  bash rig_calibrate.sh /workspace/rigtest [sync_frame_idx]
set -euo pipefail

VID_DIR=${1:?usage: rig_calibrate.sh <vid_dir> [frame_idx]}
FRAME=${2:-30}                       # avoid frame 0 (exposure settle)
REPO=${REPO:-/workspace/my-4d-video-project/lifecast_apps}
SFM=$REPO/bazel-bin/source/incremental_sfm
BASELINE=${BASELINE:-0.060}
CAMS=(cam0L cam0R cam1L cam1R cam2L cam2R)
CAL=$VID_DIR/calib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/torch/lib

[ -x "$SFM" ] || { echo "FATAL: $SFM not built. bazel build -c opt //source:incremental_sfm"; exit 2; }

rm -rf "$CAL"; mkdir -p "$CAL/frames"

echo "=== [1/3] extract sync'd frame $FRAME from each eye ==="
for c in "${CAMS[@]}"; do
  ffmpeg -y -hide_banner -loglevel error -i "$VID_DIR/$c.mp4" \
    -vf "select=eq(n\,$FRAME)" -vframes 1 "$CAL/frames/$c.png"
  [ -s "$CAL/frames/$c.png" ] || { echo "FATAL: no frame $FRAME in $c.mp4"; exit 1; }
done
# 6-frame sequence, order = CAMS order so cameras 0,1 are cam0L,cam0R
# (that pair's distance is what --dist_a_to_b pins to 60 mm)
i=0; for c in "${CAMS[@]}"; do cp "$CAL/frames/$c.png" "$CAL/frames/cam${i}_seq.png"; i=$((i+1)); done
ffmpeg -y -hide_banner -loglevel error -framerate 1 \
  -i "$CAL/frames/cam%d_seq.png" -c:v libx264 -qp 0 -pix_fmt yuv444p "$CAL/seq.mp4"

echo "=== [2/3] fisheye SfM (metric via ${BASELINE}m intra-pair baseline) ==="
# --subsample 1: keep all 6 frames.  --time_window_size 0: match all pairs.
# --max_image_dim: SfM rescales intrinsics with the image, and the merge step
# below rescales them back to full resolution, so 1280 is fine and fast.
"$SFM" --src_vid "$CAL/seq.mp4" --dest_dir "$CAL" \
  --fisheye --dist_a_to_b "$BASELINE" --subsample 1 --time_window_size 0 \
  --share_intrinsics --intrinsic_prior --max_image_dim 1280 \
  --inlier_frac 0.8 --max_solver_itrs 200
[ -s "$CAL/dataset.json" ] || { echo "FATAL: SfM wrote no dataset.json"; exit 1; }

echo "=== [3/3] merge refined poses back into $VID_DIR/dataset.json ==="
python3 - "$VID_DIR" "$CAL" <<'PYEOF'
import json, sys, shutil
vid_dir, cal = sys.argv[1], sys.argv[2]
CAMS = ["cam0L","cam0R","cam1L","cam1R","cam2L","cam2R"]

rig = json.load(open(f"{vid_dir}/dataset.json"))
sfm = json.load(open(f"{cal}/dataset.json"))
sf = sfm["frames_data"]
assert len(sf) == 6, f"SfM registered {len(sf)}/6 cameras - inspect {cal}"

# SfM frames are in CAMS order (that is how seq.mp4 was assembled).
by_name = {fr["image_filename"]: fr for fr in rig["frames_data"]}
for i, sfr in enumerate(sf):
    tgt = by_name[CAMS[i]]
    # rescale intrinsics from SfM resolution back to full per-eye resolution
    s = tgt["width"] / float(sfr["width"])
    tgt["world_from_cam"] = sfr["world_from_cam"]
    for k in ("radius_at_90", "useable_radius", "cx", "cy"):
        if k in sfr: tgt[k] = sfr[k] * s
    for k in ("k1", "tilt"):
        if k in sfr: tgt[k] = sfr[k]
rig["_note"] = f"poses+intrinsics refined by rig_calibrate.sh from {cal}"

shutil.copy(f"{vid_dir}/dataset.json", f"{vid_dir}/dataset.seed.json")
json.dump(rig, open(f"{vid_dir}/dataset.json", "w"), indent=2)

# quick metric sanity: report the three intra-pair baselines
import math
def pos(fr):
    m = fr["world_from_cam"]  # column-major flat 16; translation = elems 12,13,14
    return (m[12], m[13], m[14])
for p in range(3):
    a, b = pos(by_name[f"cam{p}L"]), pos(by_name[f"cam{p}R"])
    d = math.dist(a, b)
    print(f"  pair {p} baseline: {d*1000:.1f} mm" + ("  <-- CHECK" if abs(d-0.060) > 0.01 else ""))
print("MERGE_DONE")
PYEOF

cp "$CAL/pointcloud_sfm.bin" "$VID_DIR/pointcloud_sfm.bin" 2>/dev/null || true
echo "RIG_CALIBRATE_DONE $VID_DIR"
