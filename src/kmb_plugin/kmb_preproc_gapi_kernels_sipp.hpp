// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
