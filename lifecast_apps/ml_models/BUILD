# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

exports_files([
  "rof_cuda.pt",
  "rof_cpu.pt",
  "edsr2_noaug.pt",
  "edsr2_mystery.pt",
  "unet_noaug.pt",
  "unet_mystery.pt",
  "depth_anything3_cuda.pt",
])

# Depth Anything 3 stereo module, exported by scripts/export_da3_torchscript.py.
# ~518 MB, so it is NOT committed - fetch it with scripts/fetch_models.sh.
# This is a glob-backed filegroup rather than an exports_files entry on purpose:
# a glob that matches nothing is legal and yields an empty filegroup, so the
# repo still builds when the model is absent. Once the file exists it is placed
# in runfiles at ml_models/da3_stereo.pt, which is the default path that
# depth_anything3.cc looks for - so no flag is needed.
filegroup(
  name = "da3_stereo",
  srcs = glob(["da3_stereo.pt"]),
  visibility = ["//visibility:public"],
)
