#!/usr/bin/env python3
"""Patch source/ldi_common.cc for static-camera background-plate inpainting.

For a locked-off shot, the true background behind a moving subject is visible in
OTHER frames of the same shot. make_plate.py recovers it as a temporal median.
This patch makes the LDI pipeline fill layer-0 disocclusions from that plate
instead of the Ceres interpolation, when LIFECAST_PLATE_PATH is set.

IMPORTANT: the plate is applied ONLY inside l0_inpaint_mask (the disoccluded
region). Everywhere else the real frame's pixels are kept. This matters for
outdoor shots: a temporal median time-averages wind-blown grass and foliage, so
using it for the whole layer would freeze/blur moving background. Inside the
holes that averaging is invisible - no viewer knows which blade of grass was
where - while the recovered content is real background rather than a smear.

Layer 1 (mid) keeps the Ceres fill: plate content there sits at the wrong depth.

Usage:  python3 patch_plate_inpaint.py [path/to/ldi_common.cc]
Then:   bazel build -c opt --cuda_archs=compute_89 //source:vve_cli
Run:    LIFECAST_PLATE_PATH=/path/plate.png vve_cli --phase inpaint ...
"""

import sys

DEFAULT = '/workspace/my-4d-video-project/lifecast_apps/source/ldi_common.cc'
ANCHOR = '  if (assemble_ldi) {\n    assembleLayersAndChannels('
PATCH = '''  // Static-camera background plate (jg stereo editor).
  // If LIFECAST_PLATE_PATH is set, fill the layer-0 disocclusions from a
  // temporal-median plate - real background pixels recovered from other frames
  // of the same locked-off shot - instead of the Ceres interpolation.
  // Applied only inside l0_inpaint_mask so that moving background (wind in
  // grass/foliage) is preserved from the actual frame everywhere else.
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
