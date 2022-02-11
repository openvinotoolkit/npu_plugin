// Copyright (C) 2019 Intel Corporation
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#if defined(__arm__) || defined(__aarch64__)
#include <opencv2/gapi.hpp>

namespace InferenceEngine {
namespace gapi {
namespace preproc {
namespace sipp {
cv::gapi::GKernelPackage kernels();
}  // namespace sipp
}  // namespace preproc
}  // namespace gapi
}  // namespace InferenceEngine
#endif
