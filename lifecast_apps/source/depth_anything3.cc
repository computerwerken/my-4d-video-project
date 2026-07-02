// MIT License. (Follows the conventions of the lifecast_public codebase.)
// See depth_anything3.h for the module contract.

#include "depth_anything3.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "logger.h"
#include "util_math.h"
#include "util_runfile.h"
#include "util_time.h"
#include "util_torch.h"

namespace p11 { namespace depth_estimation {

namespace {

torch::DeviceType findDa3Device()
{
  torch::DeviceType device = util_torch::findBestTorchDevice();
  if (const char* force_cpu = std::getenv("LIFECAST_FORCE_CPU")) {
    if (std::string(force_cpu) == "1") device = torch::kCPU;
  }
  return device;
}

// BGR(A) 8U -> RGB float [0,1] resized to kDa3InputSize, as a [1, 3, H, W] tensor.
torch::Tensor preprocessOneView(const cv::Mat& image_bgr)
{
  cv::Mat rgb;
  if (image_bgr.type() == CV_8UC3) {
    cv::cvtColor(image_bgr, rgb, cv::COLOR_BGR2RGB);
  } else if (image_bgr.type() == CV_8UC4) {
    cv::cvtColor(image_bgr, rgb, cv::COLOR_BGRA2RGB);
  } else {
    XCHECK(false) << "unexpected image type for DA3: " << image_bgr.type();
  }
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(kDa3InputSize, kDa3InputSize), 0, 0, cv::INTER_CUBIC);
  resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

  torch::Tensor t =
      torch::from_blob(resized.data, {1, kDa3InputSize, kDa3InputSize, 3}, torch::kFloat);
  // clone() before `resized` goes out of scope; permute to NCHW.
  return t.permute({0, 3, 1, 2}).clone();
}

// [H, W] float tensor (any device/dtype) -> CV_32FC1 Mat.
cv::Mat tensorToMat32F(torch::Tensor t)
{
  t = t.to(torch::kCPU).to(torch::kFloat).contiguous();
  XCHECK_EQ(t.dim(), 2);
  cv::Mat m(cv::Size((int)t.size(1), (int)t.size(0)), CV_32FC1, t.data_ptr<float>());
  return m.clone();  // clone so the Mat owns its memory after the tensor dies
}

}  // namespace

void getTorchModelDepthAnything3(torch::jit::script::Module& module, std::string model_path)
{
  torch::NoGradGuard no_grad;

#if defined(__APPLE__)
  // Fall back to CPU for any op not yet implemented on MPS instead of crashing.
  setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1", /*overwrite=*/0);
#endif

  if (model_path.empty()) {
#if defined(_WIN32)
    const std::string default_model = "da3_stereo.pt";  // flat directory structure on windows
#else
    const std::string default_model = "ml_models/da3_stereo.pt";
#endif
    model_path = p11::runfile::getRunfileResourcePath(default_model);
  }
  XPLINFO << "DA3 model_path: " << model_path;

  try {
    module = torch::jit::load(model_path);
    module.eval();
  } catch (const c10::Error& e) {
    XCHECK(false) << "Error loading DA3 torch module: " << e.what() << "\n" << e.msg();
  }
}

DA3StereoResult estimateInvDepthDA3Stereo(
    torch::jit::script::Module& module,
    const cv::Mat& L_image_bgr,
    const cv::Mat& R_image_bgr)
{
  torch::NoGradGuard no_grad;
  XCHECK_EQ(L_image_bgr.size(), R_image_bgr.size());

  const torch::DeviceType device = findDa3Device();

  // Stack views: [2, 3, H, W]. Index 0 = left, 1 = right (matches the export wrapper).
  torch::Tensor input =
      torch::cat({preprocessOneView(L_image_bgr), preprocessOneView(R_image_bgr)}, 0).to(device);

  module.to(device);
  auto output = module.forward({input});
  auto tup = output.toTuple();
  torch::Tensor depth = tup->elements()[0].toTensor();  // [2, H, W]
  torch::Tensor conf = tup->elements()[1].toTensor();   // [2, H, W]
  XCHECK_EQ(depth.dim(), 3);
  XCHECK_EQ(depth.size(0), 2);

  // Right view only. DA3 outputs depth; the pipeline works in inverse depth.
  torch::Tensor r_depth = depth[1];
  torch::Tensor r_inv = 1.0 / torch::clamp(r_depth, 1e-6);

  // Min-max normalize confidence to [0, 1] per frame (DA3 conf scale is model-dependent).
  torch::Tensor r_conf = conf[1];
  const float conf_min = r_conf.min().item<float>();
  const float conf_max = r_conf.max().item<float>();
  r_conf = (r_conf - conf_min) / std::max(conf_max - conf_min, 1e-6f);

  DA3StereoResult result;
  result.inv_depth = tensorToMat32F(r_inv);
  result.confidence = tensorToMat32F(r_conf);

  // Resize back to the caller's image size (same convention as the RAFT path).
  cv::resize(result.inv_depth, result.inv_depth, R_image_bgr.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(result.confidence, result.confidence, R_image_bgr.size(), 0, 0, cv::INTER_LINEAR);
  return result;
}

cv::Mat fuseWithStereoInvDepth(
    const cv::Mat& da3_inv_depth,
    const cv::Mat& da3_conf,
    const cv::Mat& stereo_inv_depth,
    float blend)
{
  XCHECK_EQ(da3_inv_depth.size(), stereo_inv_depth.size());
  XCHECK_EQ(da3_inv_depth.size(), da3_conf.size());
  XCHECK_EQ(da3_inv_depth.type(), CV_32FC1);
  XCHECK_EQ(stereo_inv_depth.type(), CV_32FC1);
  XCHECK_EQ(da3_conf.type(), CV_32FC1);

  blend = math::clamp(blend, 0.0f, 1.0f);
  if (blend <= 0.0f) return stereo_inv_depth.clone();

  const cv::Mat& x = da3_inv_depth;     // relative inv depth (to be aligned)
  const cv::Mat& y = stereo_inv_depth;  // metric inv depth (alignment target)

  // Robust weighted least squares for scale s and shift t minimizing
  // sum w * (s*x + t - y)^2, with IRLS reweighting to reject outliers
  // (occlusions, spec highlights, regions where either model fails).
  cv::Mat w;
  cv::max(da3_conf, 1e-3f, w);

  double s = 1.0, t = 0.0;
  for (int iter = 0; iter < 3; ++iter) {
    const double Sw = cv::sum(w)[0];
    const double Swx = cv::sum(w.mul(x))[0];
    const double Swy = cv::sum(w.mul(y))[0];
    const double Swxx = cv::sum(w.mul(x.mul(x)))[0];
    const double Swxy = cv::sum(w.mul(x.mul(y)))[0];

    const double det = Sw * Swxx - Swx * Swx;
    if (std::abs(det) < 1e-12) {
      XPLINFO << "DA3 alignment: degenerate system, keeping stereo depth";
      return stereo_inv_depth.clone();
    }
    s = (Sw * Swxy - Swx * Swy) / det;
    t = (Swxx * Swy - Swx * Swxy) / det;

    // Reweight by residual (Cauchy-style), keeping confidence as a factor.
    cv::Mat residual;
    cv::absdiff(x * s + t, y, residual);
    const double sigma = std::max(cv::mean(residual)[0] * 1.2533, 1e-6);  // ~= robust std
    cv::Mat residual_sq = residual.mul(residual) * (1.0 / (4.0 * sigma * sigma));
    cv::Mat denom = residual_sq + 1.0;
    cv::Mat irls_weight;
    cv::divide(1.0, denom, irls_weight);
    cv::max(da3_conf, 1e-3f, w);
    w = w.mul(irls_weight);
  }
  XPLINFO << "DA3 alignment: scale=" << s << " shift=" << t;

  // A negative or tiny scale means the fit failed (e.g. stereo depth was degenerate).
  if (s <= 1e-9) {
    XPLINFO << "DA3 alignment: non-positive scale, keeping stereo depth";
    return stereo_inv_depth.clone();
  }

  cv::Mat aligned = x * s + t;
  cv::max(aligned, 0.0f, aligned);

  // Confidence-weighted blend: alpha = blend * conf, fused = alpha*da3 + (1-alpha)*stereo.
  cv::Mat alpha = da3_conf * blend;
  cv::Mat fused = alpha.mul(aligned) + (cv::Scalar::all(1.0) - alpha).mul(y);
  cv::max(fused, 0.0f, fused);
  return fused;
}

}}  // namespace p11::depth_estimation
