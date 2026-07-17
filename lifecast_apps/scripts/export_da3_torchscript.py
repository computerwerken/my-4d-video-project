#!/usr/bin/env python3
# Export Depth Anything 3 (DA3-BASE) as a fixed-shape 2-view TorchScript module
# matching source/depth_anything3.cc's contract:
#   input:  [2, 3, 1036, 1036] float32 RGB in [0,1]  (index 0 = left, 1 = right)
#   output: tuple(depth [2,H,W], confidence [2,H,W])
# ImageNet normalization and the [1,2,3,H,W] view-batch reshape are baked inside.
#
# IMPORTANT: run with torch matching the C++ libtorch version (2.5.1 as of this
# commit). Validated 2026-07-17 on RunPod RTX 4090: trace-vs-eager corr 1.0,
# real-stereo-pair corr vs official DepthAnything3.inference() = 0.96+.
#
# Setup:
#   python3 -m venv da3env
#   da3env/bin/pip install torch==2.5.1 torchvision==0.20.1 \
#       --index-url https://download.pytorch.org/whl/cu124
#   da3env/bin/pip install depth-anything-3 && da3env/bin/pip install -U numpy
#   da3env/bin/python export_da3_torchscript.py

import torch
from depth_anything_3.api import DepthAnything3

OUT = 'da3_stereo.pt'
SIZE = 1036  # must be a multiple of 14 (DINOv2 patch size)


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('m', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer('s', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, x):
        x5 = (x.unsqueeze(0) - self.m) / self.s
        out = self.net(x5, None, None, [], False, False, 'saddle_balanced')
        return out['depth'][0], out['depth_conf'][0]


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mm = DepthAnything3.from_pretrained('depth-anything/DA3-BASE')
    w = Wrapper(mm.model).to(dev).eval()
    x = torch.rand(2, 3, SIZE, SIZE, device=dev)
    with torch.no_grad():
        rd, rc = w(x)
        tr = torch.jit.trace(w, x, strict=False)
        td, tc = tr(x)
    c1 = torch.corrcoef(torch.stack([rd.flatten(), td.flatten()]))[0, 1].item()
    print('trace corr', round(c1, 6), 'maxdiff', (rd - td).abs().max().item())
    assert c1 > 0.999, 'trace does not match eager execution'
    tr.save(OUT)
    r = torch.jit.load(OUT).eval()
    with torch.no_grad():
        d2, _ = r(x)
    c2 = torch.corrcoef(torch.stack([rd.flatten(), d2.flatten()]))[0, 1].item()
    assert c2 > 0.999, 'saved module does not reload faithfully'
    print('EXPORT_DONE ->', OUT)


if __name__ == '__main__':
    main()
