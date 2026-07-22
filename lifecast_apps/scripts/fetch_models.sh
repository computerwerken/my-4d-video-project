#!/usr/bin/env bash
# Fetch the model files that are not stored in git.
#
#   ml_models/da3_stereo.pt   ~518 MB, GitHub Release asset (too big for LFS)
#   ml_models/rof_*.pt        git-lfs; this script only verifies they were pulled
#
# Usage:
#   ./scripts/fetch_models.sh              # fetch anything missing
#   ./scripts/fetch_models.sh --force      # re-download even if present
#   DA3_MODEL_URL=... ./scripts/fetch_models.sh   # override the source URL
#
# Exit codes: 0 ok, 1 fetch failed, 2 environment problem.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML="$REPO_ROOT/ml_models"
DA3="$ML/da3_stereo.pt"

# Override this if you host the model somewhere else.
DEFAULT_URL="https://github.com/computerwerken/my-4d-video-project/releases/download/models-v1/da3_stereo.pt"
URL="${DA3_MODEL_URL:-$DEFAULT_URL}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# A valid TorchScript archive is a zip, so it starts with the bytes "PK".
# Checking this catches the two failure modes that actually happen: an HTML
# error page saved as .pt, and a truncated download.
is_torchscript() {
  [ -f "$1" ] || return 1
  [ "$(head -c 2 "$1" 2>/dev/null)" = "PK" ] || return 1
  # anything under 100 MB is certainly not DA3-BASE
  local sz
  sz=$(wc -c < "$1")
  [ "$sz" -gt 100000000 ] || return 1
}

echo "repo root: $REPO_ROOT"

# --- 1. git-lfs sanity ------------------------------------------------------
# If LFS was never pulled, the .pt files are ~130-byte pointer files and every
# model load fails with a confusing torch error. Catch that here instead.
# NB: plain string accumulator, not a bash array. macOS still ships bash 3.2,
# where `${#arr[@]}` on an empty array trips `set -u`.
lfs_pointers=""
for f in "$ML"/rof_cpu.pt "$ML"/rof_cuda.pt; do
  [ -f "$f" ] || continue
  if head -c 40 "$f" 2>/dev/null | grep -q "version https://git-lfs"; then
    lfs_pointers="$lfs_pointers $(basename "$f")"
  fi
done
if [ -n "$lfs_pointers" ]; then
  echo "ERROR: these are git-lfs pointer files, not real models:" >&2
  for p in $lfs_pointers; do echo "   $p" >&2; done
  echo "Run:  git lfs install && git lfs pull" >&2
  exit 2
fi
echo "git-lfs models: OK"

# --- 2. DA3 module ----------------------------------------------------------
if [ "$FORCE" -eq 0 ] && is_torchscript "$DA3"; then
  echo "da3_stereo.pt: already present ($(du -h "$DA3" | cut -f1)), skipping"
  exit 0
fi

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found" >&2; exit 2; }

echo "fetching da3_stereo.pt"
echo "   from: $URL"
TMP="$DA3.part"
rm -f "$TMP"

# --fail so an HTTP error is an error rather than a saved error page;
# -C - to resume a partial download rather than restarting 518 MB.
if ! curl -fL --retry 3 --retry-delay 2 -C - -o "$TMP" "$URL"; then
  rm -f "$TMP"
  cat >&2 <<EOF

ERROR: download failed.

If this was a 404, the release asset probably has not been published yet.
To publish it from a machine that has a working da3_stereo.pt:

    gh release create models-v1 ml_models/da3_stereo.pt \\
        --title "Model assets v1" \\
        --notes "DA3-BASE traced for 2-view stereo (Apache-2.0)."

Or export the model locally instead (needs a GPU and ~10 min):

    python3 scripts/export_da3_torchscript.py && mv da3_stereo.pt ml_models/

Or set DA3_MODEL_URL to wherever you have it hosted.
EOF
  exit 1
fi

if ! is_torchscript "$TMP"; then
  echo "ERROR: downloaded file is not a valid TorchScript archive." >&2
  echo "   size: $(wc -c < "$TMP") bytes; first bytes: $(head -c 2 "$TMP" | od -c | head -1)" >&2
  echo "   (a login/404 HTML page saved as .pt looks exactly like this)" >&2
  rm -f "$TMP"
  exit 1
fi

mv -f "$TMP" "$DA3"
echo "da3_stereo.pt: OK ($(du -h "$DA3" | cut -f1))"

if command -v shasum >/dev/null 2>&1; then
  echo "   sha256: $(shasum -a 256 "$DA3" | cut -d' ' -f1)"
fi

cat <<EOF

Done. Rebuild so the model is picked up in runfiles:
    bazel build -c opt //source:vve_cli
Then --depth_method=da3_fused works with no --da3_model_path flag.
EOF
