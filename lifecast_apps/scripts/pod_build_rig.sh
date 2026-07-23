#!/usr/bin/env bash
# Build + verify the 3xR5C rig demo binaries on the RunPod 4090
# (see RIG_PIPELINE.md). Companion to pod_build.sh, with its two hard-won
# fixes baked in: USE_BAZEL_VERSION must be exported (bazelisk otherwise
# grabs Bazel 9, which ignores WORKSPACE entirely), and the persisted cache
# is an --output_user_root, not an --output_base.
#
# Builds:
#   //source:incremental_sfm   (with the new --fisheye / --dist_a_to_b flags)
#   //source:lifecast_splat    (gsplat CUDA kernels: NOT in the cache yet,
#                               expect the long pole here, ~15-40 min)
#
# Run detached; the web terminal truncates foreground output:
#   curl -fsSL https://raw.githubusercontent.com/computerwerken/my-4d-video-project/jg-stereo-editor/lifecast_apps/scripts/pod_build_rig.sh -o /workspace/pod_build_rig.sh
#   setsid bash /workspace/pod_build_rig.sh > /workspace/rig_build.log 2>&1 & disown
#   tail -f /workspace/rig_build.log
set -uo pipefail

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
export USE_BAZEL_VERSION=7.2.0
export DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/torch/lib
CUDA_ARCHS=${CUDA_ARCHS:-compute_89}
CACHE=${BAZEL_OUTPUT_USER_ROOT:-/workspace/bazel_out}
REPO=${REPO:-/workspace/my-4d-video-project}
APP="$REPO/lifecast_apps"

log "=== pod_build_rig start ==="

# --- environment: reuse the proven restore path -------------------------------
if [ -f /workspace/restore.sh ] && [ ! -f /usr/local/torch/lib/libtorch.so ]; then
  log "container root was reset; running restore.sh"
  bash /workspace/restore.sh || { log "FATAL restore.sh"; exit 2; }
fi
command -v bazel >/dev/null || { log "FATAL: bazel missing even after restore"; exit 2; }
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' || log "WARN: nvidia-smi failed"

# --- sync repo ----------------------------------------------------------------
cd "$APP" || { log "FATAL cd $APP"; exit 2; }
git -C "$REPO" fetch --depth 1 origin jg-stereo-editor 2>&1 | tail -1
git -C "$REPO" reset --hard origin/jg-stereo-editor 2>&1 | tail -1
log "repo HEAD: $(git -C "$REPO" rev-parse --short HEAD)"
chmod +x "$APP"/scripts/*.sh 2>/dev/null || true

# --- build both targets -------------------------------------------------------
fail=0
for target in //source:incremental_sfm //source:lifecast_splat; do
  log "=== bazel build $target ==="
  if bazel --output_user_root="$CACHE" build -c opt --cuda_archs="$CUDA_ARCHS" "$target" 2>&1 | tail -20; then
    log "build ok: $target"
  else
    log "RESULT: FAIL building $target"; fail=$((fail+1))
  fi
done
[ "$fail" -eq 0 ] || { log "RESULT: HAS_FAILURES ($fail)"; exit 1; }

# --- smoke checks against the built binaries ----------------------------------
log "=== smoke checks ==="
SFM="$APP/bazel-bin/source/incremental_sfm"
SPLAT="$APP/bazel-bin/source/lifecast_splat"
pass=0; miss=0
chk(){ if "$1" --help 2>&1 | grep -q -- "$2"; then echo "  PASS  $3"; pass=$((pass+1)); else echo "  FAIL  $3"; miss=$((miss+1)); fi; }
chk "$SFM"   "-fisheye"      "incremental_sfm has --fisheye"
chk "$SFM"   "-dist_a_to_b"  "incremental_sfm has --dist_a_to_b"
chk "$SFM"   "-subsample"    "incremental_sfm has --subsample"
chk "$SPLAT" "-vid_dir"      "lifecast_splat has --vid_dir (video/rig mode)"
chk "$SPLAT" "-init_with_monodepth" "lifecast_splat has --init_with_monodepth"

log "=== RESULT: $pass passed, $miss failed ==="
if [ "$miss" -eq 0 ]; then
  log "RESULT: ALL_GREEN - ready for footage. Next:"
  log "  1. upload the three 10s Premiere exports to /workspace/"
  log "  2. python3 $APP/scripts/rig_prepare.py A.mov B.mov C.mov --out /workspace/rigtest --dur 10"
  log "  3. bash $APP/scripts/rig_calibrate.sh /workspace/rigtest"
  log "  4. $SPLAT --vid_dir /workspace/rigtest --output_dir /workspace/rigout"
else
  log "RESULT: HAS_FAILURES"
fi
exit "$miss"
