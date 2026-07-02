// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// MODIFIED: MPS (Metal) inference enabled on Apple Silicon.
//   - Removed the hard-coded `device = torch::kCPU` hack in computeOpticalFlowRAFT.
//   - PYTORCH_ENABLE_MPS_FALLBACK=1 is set on Apple so ops missing on MPS fall back
//     to CPU instead of crashing (the original reason for the hack).
//   - Escape hatch: run with LIFECAST_FORCE_CPU=1 to restore old CPU-only behavior.
//   - Replaced the per-pixel tensor->Mat copy loop with a bulk 2-channel copy.
//   - Vectorized the bias/clamp and error-map loops.

#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <regex>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "rof.h"
#include "torch/torch.h"
#include "logger.h"
#include "util_runfile.h"
#include "util_time.h"
#include "util_torch.h"

namespace p11 { namespace optical_flow {

namespace {
// Returns the device to run flow inference on, honoring the LIFECAST_FORCE_CPU=1
// escape hatch (useful if a particular model/libtorch combo misbehaves on MPS).
torch::DeviceType findFlowDevice()
{
  torch::DeviceType device = util_torch::findBestTorchDevice();
  if (const char* force_cpu = std::getenv("LIFECAST_FORCE_CPU")) {
    if (std::string(force_cpu) == "1") device = torch::kCPU;
  }
  return device;
}
}  // namespace

void getTorchModelRAFT(torch::jit::script::Module& module, std::string model_path)
{
  torch::NoGradGuard no_grad;

#if defined(__APPLE__)
  // Allow torch to fall back to CPU for any op not yet implemented on MPS
  // (e.g. the old aten::searchsorted gap). Must be set before the first MPS op
  // dispatch. overwrite=0 respects a value the user already exported.
  setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1", /*overwrite=*/0);
#endif

  if (model_path.empty()) {
#if defined(__linux__)
    const std::string default_model = "ml_models/rof_cuda.pt";
#elif defined(_WIN32)
    const std::string default_model = "rof_cuda.pt";  // On windows the directory structure is flat
#elif defined(__APPLE__)
    // rof_cpu.pt is a CPU-traced TorchScript module; with a modern libtorch it can be
    // moved to MPS via .to(). If that trips on baked-in CPU device constants, re-export
    // with scripts/export_raft_torchscript.py --device mps and use ml_models/rof_mps.pt.
    const std::string default_model = "ml_models/rof_cpu.pt";
#else
    const std::string default_model = "ml_models/rof_cpu.pt";
#endif
    model_path = p11::runfile::getRunfileResourcePath(default_model);
  }
  XPLINFO << "model_path: " << model_path;

  try {
#ifdef _WIN32
    torch::DeviceType device = util_torch::findBestTorchDevice();
    XCHECK_EQ(device, torch::kCUDA) << "CUDA not avilable, but required for Windows";
    std::ifstream inp(model_path, std::ios::binary);
    XCHECK(inp.is_open()) << "Failed to open ML model file: " << model_path;
    module = torch::jit::load(inp, device);
#else
    module = torch::jit::load(model_path);
#endif
    module.eval();
  } catch (const c10::Error& e) {
    XCHECK(false) << "Error loading torch module: " << e.what() << "\n" << e.msg();
  }
}

void computeOpticalFlowRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    cv::Mat& flow_x,
    cv::Mat& flow_y)
{
  torch::NoGradGuard no_grad;

  XCHECK_EQ(image1_8u.size(), image2_8u.size());
  XCHECK(image1_8u.type() == image2_8u.type());
  XCHECK(image1_8u.type() == CV_8UC3 || image1_8u.type() == CV_8UC4);
  XCHECK(image2_8u.type() == CV_8UC3 || image2_8u.type() == CV_8UC4);
  XCHECK_EQ(image1_8u.rows % 8, 0);
  XCHECK_EQ(image1_8u.cols % 8, 0);
  XCHECK_EQ(image2_8u.rows % 8, 0);
  XCHECK_EQ(image2_8u.cols % 8, 0);

  cv::Mat image1, image2;
  // The original python implementation expects RGB, not BGR
  if (image1_8u.type() == CV_8UC3) {
    cv::cvtColor(image1_8u, image1, cv::COLOR_BGR2RGB);
    cv::cvtColor(image2_8u, image2, cv::COLOR_BGR2RGB);
  } else if (image1_8u.type() == CV_8UC4) {
    cv::cvtColor(image1_8u, image1, cv::COLOR_BGRA2RGB);
    cv::cvtColor(image2_8u, image2, cv::COLOR_BGRA2RGB);
  } else {
    XCHECK(false) << "unexpected image type";
  }
  image1.convertTo(image1, CV_32FC3);
  image2.convertTo(image2, CV_32FC3);

  // Pack the image data into a tensor with appropriate shape.
  at::Tensor tensor_image1 =
      torch::from_blob(image1.data, {image1.rows, image1.cols, 3}, at::kFloat);
  at::Tensor tensor_image2 =
      torch::from_blob(image2.data, {image2.rows, image2.cols, 3}, at::kFloat);

  // MPS is used on Apple Silicon when available (previously forced to CPU here).
  const torch::DeviceType device = findFlowDevice();

  tensor_image1.unsqueeze_(0);
  tensor_image2.unsqueeze_(0);
  auto tensor_image1_perm = tensor_image1.permute({0, 3, 1, 2});
  auto tensor_image2_perm = tensor_image2.permute({0, 3, 1, 2});

  std::vector<torch::jit::IValue> inputs;
  auto inp1 = tensor_image1_perm.to(device);
  auto inp2 = tensor_image2_perm.to(device);
  inputs.push_back(inp1);
  inputs.push_back(inp2);

  module.to(device);

  auto rawOutputs = module.forward(inputs);
  auto outputs = rawOutputs.toTuple();
  auto flow_up = outputs->elements()[1].toTensor();

  auto flow_up_perm = flow_up.permute({0, 2, 3, 1});

  const int w = flow_up_perm.sizes()[2];
  const int h = flow_up_perm.sizes()[1];

  // Bring back to CPU as contiguous float32 (also handles fp16 models).
  auto flow_up_perm_cpu = flow_up_perm.to(torch::kCPU).to(torch::kFloat).contiguous();

  // Bulk copy: wrap the [1, H, W, 2] tensor as a 2-channel Mat and split.
  // cv::split copies the data, so lifetime is safe after this scope.
  cv::Mat flow_xy(cv::Size(w, h), CV_32FC2, flow_up_perm_cpu.data_ptr<float>());
  std::vector<cv::Mat> flow_channels;
  cv::split(flow_xy, flow_channels);
  flow_x = flow_channels[0];
  flow_y = flow_channels[1];
}

cv::Mat computeDisparityRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    const float bias)
{
  cv::Mat flow_x, flow_y;
  computeOpticalFlowRAFT(module, image1_8u, image2_8u, flow_x, flow_y);

  // Vectorized replacement for the old per-pixel loop:
  // disparity = max(0, flow_x + bias)
  flow_x += bias;
  cv::max(flow_x, 0.0f, flow_x);
  return flow_x;
}

void computeDisparityRAFTBothWays(
    torch::jit::script::Module& module,
    const cv::Mat& R_image,
    const cv::Mat& L_image,
    const float bias,
    cv::Mat& R_disparity,
    cv::Mat& L_disparity,
    cv::Mat& R_error,
    cv::Mat& L_error)
{
  XCHECK_EQ(R_image.size(), L_image.size());
  const int w = R_image.cols;
  const int h = R_image.rows;

  cv::Mat R_flow_x, R_flow_y;
  cv::Mat L_flow_x, L_flow_y;

  computeOpticalFlowRAFT(module, R_image, L_image, R_flow_x, R_flow_y);
  computeOpticalFlowRAFT(module, L_image, R_image, L_flow_x, L_flow_y);

  static constexpr float kVerticalDisparityErrorCoef =
      30.0;  // weight of vertical disparity in overall error
  static constexpr float kDisparityConsistencyErrorCoef = 100.0;

  // Vectorized replacements for the old per-pixel loops.
  R_flow_x += bias;
  cv::max(R_flow_x, 0.0f, R_flow_x);
  L_flow_x *= -1.0f;
  L_flow_x += bias;
  cv::max(L_flow_x, 0.0f, L_flow_x);

  // Any vertical flow suggests depth estimation / calibration error (or at least
  // uncertainty in depth).
  R_error = cv::abs(R_flow_y) * (kVerticalDisparityErrorCoef / h);
  L_error = cv::abs(L_flow_y) * (kVerticalDisparityErrorCoef / h);

  R_disparity = R_flow_x;
  L_disparity = L_flow_x;

  // Compute error estimate using loop consistency by warping, and from vertical disparity
  // TODO: idea- propage high uncertainty outward to better cover edges

  cv::Mat warp_R_from_L(cv::Size(w, h), CV_32FC2);
  cv::Mat warp_L_from_R(cv::Size(w, h), CV_32FC2);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      warp_R_from_L.at<cv::Vec2f>(y, x) =
          cv::Vec2f(x, y) + cv::Vec2f(R_disparity.at<float>(y, x), 0.0);
      warp_L_from_R.at<cv::Vec2f>(y, x) =
          cv::Vec2f(x, y) - cv::Vec2f(L_disparity.at<float>(y, x), 0.0);
    }
  }
  std::vector<cv::Mat> warp_R_from_L_uv, warp_L_from_R_uv;
  cv::split(warp_R_from_L, warp_R_from_L_uv);
  cv::split(warp_L_from_R, warp_L_from_R_uv);

  cv::Mat R_reconstructed_from_L, L_reconstructed_from_R;
  cv::remap(
      L_disparity,
      R_reconstructed_from_L,
      warp_R_from_L_uv[0],
      warp_R_from_L_uv[1],
      cv::INTER_LINEAR,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));
  cv::remap(
      R_disparity,
      L_reconstructed_from_R,
      warp_L_from_R_uv[0],
      warp_L_from_R_uv[1],
      cv::INTER_LINEAR,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));

  // Vectorized: error += |reconstructed - disparity| * coef, clamped to [0, 1].
  cv::Mat R_consistency, L_consistency;
  cv::absdiff(R_reconstructed_from_L, R_disparity, R_consistency);
  cv::absdiff(L_reconstructed_from_R, L_disparity, L_consistency);
  R_error += R_consistency * (kDisparityConsistencyErrorCoef / w);
  L_error += L_consistency * (kDisparityConsistencyErrorCoef / w);
  cv::min(R_error, 1.0f, R_error);
  cv::min(L_error, 1.0f, L_error);
  cv::max(R_error, 0.0f, R_error);
  cv::max(L_error, 0.0f, L_error);
}

}}  // namespace p11::optical_flow
