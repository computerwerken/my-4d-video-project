#!/usr/bin/env bash
# Repair an EXISTING LDI3 .mp4 into a correctly-tagged, browser-ready file.
#
# Fixes both problems that make an already-rendered clip play dark / not at all:
#   1. 4:4:4 -> 4:2:0        browsers only decode 4:2:0 (a 4:4:4 clip is a black
#                            screen in Chrome); depth lives in luma, which 4:2:0
#                            keeps full-res, so the packed depth survives.
#   2. colour range + matrix libx264 encoded the frames without any colour tags,
#                            so players guess the range and the levels land dark.
#                            We detect the real range and emit FULL-range BT.709,
#                            explicitly tagged, matching the Premiere source.
#
# This is a rescue for clips already rendered with the old (untagged) pipeline.
# For anything rendered fresh, render_with_plate.sh / finish_plate.sh now emit a
# correct *_web420.mp4 directly and you don't need this.
#
# Usage:
#   bash to_web_h264.sh input.mp4 [output.mp4]      # default output: clip.mp4
#   ASSUME_RANGE=full bash to_web_h264.sh in.mp4    # override the range guess
#
# Naming the output clip.mp4 lets the Desktop launcher auto-load it; copy the
# sidecar to clip.json alongside it and the player auto-loads that too.
set -euo pipefail

IN="${1:?usage: to_web_h264.sh input.mp4 [output.mp4]}"
OUT="${2:-clip.mp4}"
[ -f "$IN" ] || { echo "no such file: $IN" >&2; exit 1; }
command -v ffmpeg  >/dev/null || { echo "ffmpeg not found"  >&2; exit 2; }
command -v ffprobe >/dev/null || { echo "ffprobe not found" >&2; exit 2; }

echo "input:  $IN"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,pix_fmt,color_range,color_space,color_primaries,color_transfer \
  -of default=noprint_wrappers=1 "$IN" || true

# Decide how to treat the input's luma range.
#  - tagged pc/full        -> already full, don't touch the levels
#  - tagged tv/limited     -> expand 16-235 -> 0-255
#  - untagged (the common libx264-from-RGB case) -> it wrote LIMITED, so expand
# Override with ASSUME_RANGE=full|limited if you know better.
RANGE_TAG=$(ffprobe -v error -select_streams v:0 -show_entries stream=color_range \
              -of csv=p=0 "$IN" 2>/dev/null | tr -d '[:space:]' || true)
if [ -n "${ASSUME_RANGE:-}" ]; then
  INRANGE="$ASSUME_RANGE"
elif [ "$RANGE_TAG" = "pc" ] || [ "$RANGE_TAG" = "full" ]; then
  INRANGE="full"
else
  INRANGE="limited"
fi
echo "input range tag: '${RANGE_TAG:-<unset>}'  ->  treating as: $INRANGE"
echo "output: $OUT  (yuv420p, full-range BT.709, tagged)"

# scale with explicit in/out range does the deterministic level conversion;
# for full->full it is a no-op on levels. Then tag the result full BT.709.
ffmpeg -y -i "$IN" \
  -vf "scale=in_range=${INRANGE}:out_range=full" \
  -c:v libx264 -crf 14 -pix_fmt yuv420p \
  -color_range pc -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
  -x264-params "fullrange=1:colorprim=bt709:transfer=bt709:colormatrix=bt709" \
  -movflags +faststart -an "$OUT"

echo "=== wrote $OUT ==="
ffprobe -v error -select_streams v:0 \
  -show_entries stream=pix_fmt,color_range,color_space,color_primaries,color_transfer \
  -of default=noprint_wrappers=1 "$OUT"
cat <<EOF

If it still looks off, the input's colour was likely damaged by an earlier
double transcode (e.g. a VideoToolbox 4:4:4->4:2:0 pass). The definitive fix is
to re-render from source with the updated render_with_plate.sh, which tags the
encode from the start.
EOF
