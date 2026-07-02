// MIT License. (Follows the conventions of the lifecast_public codebase.)
//
// Depth Anything 3 (DA3) stereo-pair inference for the VVE LDI pipeline.
//
// DA3 is an any-view model: given both rectified views of a stereo pair it produces
// cross-view-consistent depth. Its depth is relative (up to scale/shift), so we align
// it to the metric inverse depth derived from stereo disparity (RAFT) before fusing.
//
// Expected TorchScript model (produced by scripts/export_da3_torchscript.py):
//   input:  float32 tensor [2, 3, H, W], RGB in [0, 1], H == W == kDa3InputSize
//           (index 0 = left view, index 1 = right view; ImageNet normalization is
//            baked INTO the exported wrapper, so the C++ side stays dumb)
//   output: tuple(depth [2, H, W] float32, conf [2, H, W] float32)

#pragma once

#include <string>

#include "opencv2/core.hpp"
#include "torch/script.h"
#include "torch/torch.h"

namespace p11 { namespace depth_estimation {

// Must be a multiple of 14 (DINOv2 patch size). Reduce (e.g. 770 or 518) if MPS
// memory pressure or per-frame time is too high on your machine.
constexpr int kDa3InputSize = 1036;

struct DA3StereoResult {
  cv::Mat inv_depth;   // CV_32FC1, right view, resized to match the input image size.
                       // RELATIVE inverse depth (needs scale alignment before use).
  cv::Mat confidence;  // CV_32FC1 in [0, 1] (min-max normalized per frame).
};

// Loads the DA3 TorchScript module. Empty model_path uses the platform default
// ml_models/da3_stereo.pt. On Apple, enables the MPS CPU-fallback env var.
void getTorchModelDepthAnything3(torch::jit::script::Module& module, std::string model_path = "");

// Runs DA3 on a rectified stereo pair. Returns relative inverse depth + confidence
// for the RIGHT view (the view the LDI pipeline is built around).
DA3StereoResult estimateInvDepthDA3Stereo(
    torch::jit::script::Module& module,
    const cv::Mat& L_image_bgr,
    const cv::Mat& R_image_bgr);

// Aligns DA3 relative inverse depth to metric stereo inverse depth with a
// confidence-weighted robust (IRLS) scale/shift fit, then blends:
//   fused = alpha * aligned_da3 + (1 - alpha) * stereo,  alpha = blend * confidence
// blend = 0 returns stereo unchanged; blend = 1 keeps DA3 structure at stereo scale.
// All three input Mats must be CV_32FC1 and the same size.
cv::Mat fuseWithStereoInvDepth(
    const cv::Mat& da3_inv_depth,
    const cv::Mat& da3_conf,
    const cv::Mat& stereo_inv_depth,
    float blend);

}}  // namespace p11::depth_estimation
