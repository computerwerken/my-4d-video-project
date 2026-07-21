#!/usr/bin/env python3
"""Patch source/ldi_common.cc for static-camera background-plate inpainting.

For a locked-off shot, the true background behind a moving subject is visible in
OTHER frames of the same shot. make_plate.py recovers it as a temporal median.
This patch makes the LDI pipeline use that plate as the layer-0 (background)
color instead of the Ceres interpolation fill, whenever the environment variable
LIFECAST_PLATE_PATH points at a plate image.

Layer 1 (mid) keeps the Ceres fill: plate content there would sit at the wrong
depth. Only the background layer is substituted.

Usage:  python3 patch_plate_inpaint.py [path/to/ldi_common.cc]
Then:   bazel build -c opt --cuda_archs=compute_89 //source:vve_cli
Run:    LIFECAST_PLATE_PATH=/path/plate.png vve_cli --phase inpaint ...
"""

import sys

DEFAULT = '/workspace/my-4d-video-project/lifecast_apps/source/ldi_common.cc'
ANCHOR = '  if (assemble_ldi) {\n    assembleLayersAndChannels('
PATCH = '''  // Static-camera background plate (jg stereo editor).
  // If LIFECAST_PLATE_PATH is set, use the temporal-median plate - real
  // background pixels recovered from other frames of the same locked-off shot -
  // as the layer-0 color instead of the Ceres interpolation fill.
  if (const char* jg_plate_path = std::getenv("LIFECAST_PLATE_PATH")) {
    cv::Mat jg_plate = cv::imread(jg_plate_path, cv::IMREAD_COLOR);
    if (!jg_plate.empty() && !inpainted_bottom.empty()) {
      if (jg_plate.size() != inpainted_bottom.size()) {
        cv::resize(jg_plate, jg_plate, inpainted_bottom.size(), 0.0, 0.0, cv::INTER_CUBIC);
      }
      if (jg_plate.type() != inpainted_bottom.type()) {
        jg_plate.convertTo(jg_plate, inpainted_bottom.type());
      }
      inpainted_bottom = jg_plate;
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
