#!/usr/bin/env python3
"""Patch source/ldi_common.cc for static-camera background-plate inpainting.

Two independent env-var hooks, both intended for LOCKED-OFF (tripod) shots:

  LIFECAST_PLATE_PATH        colour plate  (8-bit BGR, f-theta size)
      Fills layer-0 disocclusion holes with real background pixels recovered
      from other frames of the same shot (temporal median), instead of the
      Ceres interpolation. Applied ONLY inside l0_inpaint_mask, so moving
      background - wind in grass and foliage - is preserved from the actual
      frame everywhere the camera could see it. Inside the holes the median's
      time-averaging is imperceptible; what matters is that the content is real
      background rather than a smear.

  LIFECAST_DEPTH_PLATE_PATH  depth plate   (16-bit gray, half f-theta size)
      Replaces the layer-0 depth entirely with a median depth plate. With a
      fixed camera the background geometry is genuinely constant, so freezing
      it removes per-frame depth wobble/breathing in the background while the
      colour layer still shows everything moving. Layer 1 (mid) keeps its
      per-frame depth, since mid-ground subjects do move.

Usage:  python3 patch_plate_inpaint.py [path/to/ldi_common.cc]
Then:   bazel build -c opt --cuda_archs=compute_89 //source:vve_cli
"""

import sys

DEFAULT = '/workspace/my-4d-video-project/lifecast_apps/source/ldi_common.cc'
ANCHOR = '  if (assemble_ldi) {\n    assembleLayersAndChannels('
PATCH = '''  // Static-camera background plates (jg stereo editor).
  // Colour: fill layer-0 disocclusions from a temporal-median plate - real
  // background pixels from other frames of the same locked-off shot - applied
  // only inside l0_inpaint_mask so wind-driven grass/foliage stays live.
  if (const char* jg_plate_path = std::getenv("LIFECAST_PLATE_PATH")) {
    cv::Mat jg_plate = cv::imread(jg_plate_path, cv::IMREAD_COLOR);
    if (!jg_plate.empty() && !inpainted_bottom.empty()) {
      if (jg_plate.size() != inpainted_bottom.size()) {
        cv::resize(jg_plate, jg_plate, inpainted_bottom.size(), 0.0, 0.0, cv::INTER_CUBIC);
      }
      if (jg_plate.type() != inpainted_bottom.type()) {
        jg_plate.convertTo(jg_plate, inpainted_bottom.type());
      }
      cv::Mat jg_mask = l0_inpaint_mask;
      if (jg_mask.size() != inpainted_bottom.size()) {
        cv::resize(jg_mask, jg_mask, inpainted_bottom.size(), 0.0, 0.0, cv::INTER_NEAREST);
      }
      jg_plate.copyTo(inpainted_bottom, jg_mask);
    }
  }
  // Depth: freeze the background geometry. A locked-off camera sees constant
  // background depth, so a median depth plate removes per-frame wobble.
  if (const char* jg_dplate_path = std::getenv("LIFECAST_DEPTH_PLATE_PATH")) {
    cv::Mat jg_dplate = cv::imread(jg_dplate_path, cv::IMREAD_UNCHANGED);
    if (!jg_dplate.empty() && !l0_depth.empty()) {
      if (jg_dplate.channels() > 1) cv::cvtColor(jg_dplate, jg_dplate, cv::COLOR_BGR2GRAY);
      double jg_scale = (jg_dplate.depth() == CV_16U) ? 1.0 / 65535.0 : 1.0 / 255.0;
      jg_dplate.convertTo(jg_dplate, CV_32F, jg_scale);
      if (jg_dplate.size() != l0_depth.size()) {
        cv::resize(jg_dplate, jg_dplate, l0_depth.size(), 0.0, 0.0, cv::INTER_LINEAR);
      }
      l0_depth = jg_dplate;
    }
  }

'''


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    src = open(path).read()
    if 'LIFECAST_PLATE_PATH' in src:
        print('ALREADY_PATCHED')
        return
    if ANCHOR not in src:
        print('ANCHOR_NOT_FOUND - ldi_common.cc changed upstream')
        sys.exit(1)
    src = src.replace(ANCHOR, PATCH + ANCHOR, 1)
    if '#include <cstdlib>' not in src:
        src = src.replace('#include "ldi_common.h"',
                          '#include <cstdlib>\n#include "ldi_common.h"', 1)
    open(path, 'w').write(src)
    print('PATCHED_OK ->', path)


if __name__ == '__main__':
    main()
