#!/usr/bin/env bash
# Self-restoring build + verify for vve_cli on a RunPod box.
#
# Idempotent: every dependency is checked before installing, so it is safe to
# run whether the container root survived a migration or was reset. It syncs the
# repo to exactly origin/jg-stereo-editor (so the pod matches what was reviewed),
# builds //source:vve_cli, then asserts the new flags and corrected defaults are
# actually present in the binary - a real compile + behaviour gate, no render.
#
# Run detached and watch the log (the web terminal truncates foreground output):
#   curl -fsSL RAW_URL/scripts/pod_build.sh -o /workspace/pod_build.sh
#   setsid bash /workspace/pod_build.sh > /workspace/pod_build.log 2>&1 & disown
#   tail -f /workspace/pod_build.log      # or re-run `tail -40` as needed
#
# Env overrides:
#   REPO=/path                     repo location (auto-detected otherwise)
#   CUDA_ARCHS=compute_89          GPU arch
#   BAZEL_OUTPUT_BASE=/workspace/bazel_out
#       reuse a persisted bazel cache so the build is incremental instead of a
#       ~45 min from-scratch rebuild. On a migrated RunPod box the container
#       root (and its default ~/.cache/bazel) is wiped, but a cache on the
#       /workspace volume survives - point at it here.
set -uo pipefail

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
CUDA_ARCHS=${CUDA_ARCHS:-compute_89}     # 4090/L40S=89, A100=80, 3090=86
BAZEL_OUTPUT_BASE=${BAZEL_OUTPUT_BASE:-}
BAZEL_STARTUP=""
[ -n "$BAZEL_OUTPUT_BASE" ] && BAZEL_STARTUP="--output_base=$BAZEL_OUTPUT_BASE"
export USE_BAZEL_VERSION=7.2.0
export DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/torch/lib

log "=== pod_build start (cuda_archs=$CUDA_ARCHS) ==="

# --- 0. locate the repo -------------------------------------------------------
REPO=${REPO:-}
if [ -z "$REPO" ]; then
  for c in /workspace/my-4d-video-project /workspace/repo /workspace/*/lifecast_apps/..; do
    [ -d "$c/lifecast_apps" ] && REPO="$c" && break
  done
fi
if [ -z "$REPO" ] || [ ! -d "$REPO/lifecast_apps" ]; then
  log "repo not found on volume; cloning fresh"
  REPO=/workspace/my-4d-video-project
  git lfs install || true
  git clone --branch jg-stereo-editor \
    https://github.com/computerwerken/my-4d-video-project.git "$REPO" || { log "FATAL clone failed"; exit 2; }
fi
log "repo: $REPO"
APP="$REPO/lifecast_apps"

# --- 1. apt deps (only if a marker header is missing) -------------------------
if [ ! -f /usr/include/opencv4/opencv2/core.hpp ] || ! command -v ffmpeg >/dev/null; then
  log "installing apt deps"
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg git git-lfs unzip python3 python3-venv python3-pip \
    build-essential \
    libeigen3-dev libglfw3-dev libglfw3 libceres-dev kdialog \
    libopencv-dev libglib2.0-dev libomp-dev libturbojpeg libturbojpeg0-dev \
    libcpprest-dev libssl-dev \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libx264-dev libx265-dev libavcodec-extra >/dev/null 2>&1 \
    && log "apt deps ok" || log "WARN apt install returned $?"
else
  log "apt deps already present"
fi

# --- 2. bazel via bazelisk ----------------------------------------------------
if ! command -v bazel >/dev/null; then
  log "installing bazelisk"
  curl -fsSL -o /usr/local/bin/bazel \
    https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel || { log "FATAL bazel install"; exit 2; }
fi
log "bazel: $(bazel version 2>/dev/null | head -1 || echo '(bazelisk will fetch 7.2.0 on first build)')"

# --- 3. libtorch 2.5.1+cu124 --------------------------------------------------
if [ ! -f /usr/local/torch/lib/libtorch.so ]; then
  log "installing libtorch 2.5.1+cu124 (~2.5 GB download)"
  curl -fsSL -o /tmp/libtorch.zip \
    "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip" \
    && unzip -q /tmp/libtorch.zip -d /usr/local && rm -f /tmp/libtorch.zip \
    && mv /usr/local/libtorch /usr/local/torch \
    && ( cd /usr/local/torch/lib
         for f in libcudart-*.so.12 libnvToolsExt-*.so.1 libnvrtc-*.so.12 libgomp-*.so.1; do
           [ -e "$f" ] && ln -sf "$f" "$(echo "$f" | sed 's/-[a-f0-9]*\.so/.so/')"
         done
         ln -sf libnvrtc-builtins.so libnvrtc-builtins.so.12 2>/dev/null || true
         ln -sf libnvrtc-builtins.so libnvrtc-builtins.so.12.4 2>/dev/null || true ) \
    && log "libtorch ok" || { log "FATAL libtorch install"; exit 2; }
else
  log "libtorch already present"
fi

# --- 4. sync repo to exactly origin/jg-stereo-editor --------------------------
cd "$APP" || { log "FATAL cd $APP"; exit 2; }
log "syncing repo to origin/jg-stereo-editor"
git -C "$REPO" fetch --depth 1 origin jg-stereo-editor 2>&1 | tail -2
git -C "$REPO" checkout -q jg-stereo-editor 2>/dev/null || git -C "$REPO" checkout -q -b jg-stereo-editor
git -C "$REPO" reset --hard origin/jg-stereo-editor 2>&1 | tail -1
HEAD_SHA=$(git -C "$REPO" rev-parse --short HEAD)
log "repo HEAD: $HEAD_SHA"
chmod +x "$APP"/scripts/*.sh 2>/dev/null || true

# --- 5. models (LFS + DA3) ----------------------------------------------------
git -C "$REPO" lfs pull 2>&1 | tail -1 || log "WARN lfs pull"
# DA3 model: prefer an existing pod copy, else the repo's, else fetch.
if [ -f /workspace/da3_stereo.pt ] && [ ! -f "$APP/ml_models/da3_stereo.pt" ]; then
  log "linking existing /workspace/da3_stereo.pt into ml_models/"
  ln -sf /workspace/da3_stereo.pt "$APP/ml_models/da3_stereo.pt"
fi
if [ -f "$APP/ml_models/da3_stereo.pt" ]; then
  log "da3 model present ($(du -h "$APP/ml_models/da3_stereo.pt" | cut -f1))"
else
  log "WARN da3_stereo.pt missing - da3_fused renders will need --da3_model_path (build still proceeds)"
fi

# --- 6. build -----------------------------------------------------------------
log "=== bazel build //source:vve_cli (this is the real test of the WORKSPACE change) ==="
[ -n "$BAZEL_STARTUP" ] && log "reusing bazel cache at $BAZEL_OUTPUT_BASE (incremental build)"
# shellcheck disable=SC2086  # BAZEL_STARTUP is intentionally word-split (empty or one flag)
if bazel $BAZEL_STARTUP build -c opt --cuda_archs="$CUDA_ARCHS" //source:vve_cli 2>&1 | tail -25; then
  BIN="$APP/bazel-bin/source/vve_cli"
  [ -x "$BIN" ] || { log "RESULT: FAIL - build reported ok but $BIN missing"; exit 1; }
  log "build ok: $BIN"
else
  log "RESULT: FAIL - bazel build failed (see above)"; exit 1
fi

# --- 7. verify the binary actually has my changes -----------------------------
log "=== verifying flags + corrected defaults in the built binary ==="
HELP="$("$BIN" --help 2>&1 || true)"
pass=0; fail=0
check(){ # desc, regex
  if echo "$HELP" | grep -Eq "$2"; then echo "  PASS  $1"; pass=$((pass+1))
  else echo "  FAIL  $1"; fail=$((fail+1)); fi
}
check "ftheta_size default 3840"          'ftheta_size .*default: 3840|default: 3840'
check "inflated_ftheta_size default 1920" 'inflated_ftheta_size.*default: 1920|default: 1920'
check "rectified_size_for_depth 1280"     'rectified_size_for_depth.*default: 1280|default: 1280'
check "inpaint_method default ceres"      'inpaint_method.*default: "ceres"|default: "ceres"'
check "seg_method default heuristic"      'seg_method.*default: "heuristic"|default: "heuristic"'
check "inpaint_dilate_radius default 25"  'inpaint_dilate_radius.*default: 25|default: 25'
check "new flag --plate_path exists"      '\-plate_path'
check "new flag --depth_plate_path exists" '\-depth_plate_path'

# grid math: point at a nonexistent src so it prints the grid line then exits.
# The corrected line reports 3 x inflated (5760), not 3 x ftheta.
log "--- grid-math smoke (expect 'LDI3 grid 5760x5760' for tile 1920) ---"
mkdir -p /tmp/vb
GRID="$("$BIN" --src_vr180 /nonexistent.mov --dest_dir /tmp/vb --ftheta_size 3840 \
        --inflated_ftheta_size 1920 --output_encoding split12 2>&1 | grep -m1 'LDI3 grid' || true)"
echo "  $GRID"
if echo "$GRID" | grep -q '5760x5760'; then echo "  PASS  grid = 3 x inflated"; pass=$((pass+1))
else echo "  FAIL  grid line wrong or absent"; fail=$((fail+1)); fi

echo
log "=== RESULT: $pass passed, $fail failed  (HEAD $HEAD_SHA) ==="
if [ "$fail" -eq 0 ]; then log "RESULT: ALL_GREEN"; else log "RESULT: HAS_FAILURES"; fi
exit "$fail"
