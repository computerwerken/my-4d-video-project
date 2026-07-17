#!/usr/bin/env python3
# Verify an exported da3_stereo.pt against the official DepthAnything3 API on a
# REAL stereo pair (the honest test; random-tensor correlation is not enough).
# Pass criterion (from the project handoff): correlation >= 0.95 per view.
# Note the official API processes at its default resolution (504), the traced
# module at 1036 - the comparison resizes, so expect ~0.96-0.98, not 1.0.
#
# Usage: python3 verify_da3_torchscript.py L.png R.png [da3_stereo.pt]

import sys

import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

SIZE = 1036


def prep(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    return torch.from_numpy(r).permute(2, 0, 1)


def main():
    l_path, r_path = sys.argv[1], sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'da3_stereo.pt'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    L = cv2.imread(l_path)
    R = cv2.imread(r_path)
    x = torch.stack([prep(L), prep(R)]).to(dev)

    tr = torch.jit.load(model_path).eval()
    with torch.no_grad():
        td, tc = tr(x)
    print('traced output', tuple(td.shape))

    m = DepthAnything3.from_pretrained('depth-anything/DA3-BASE').to(dev).eval()
    pred = m.inference([l_path, r_path])
    od = np.asarray(pred.depth)
    print('official output', od.shape)

    ok = True
    for v in (0, 1):
        o = cv2.resize(od[v].astype(np.float32), (SIZE, SIZE))
        t = td[v].float().cpu().numpy()
        c = np.corrcoef(o.flatten(), t.flatten())[0, 1]
        print('view %d corr: %.4f' % (v, c))
        ok = ok and c >= 0.95
    print('VERIFY_PASS' if ok else 'VERIFY_FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
