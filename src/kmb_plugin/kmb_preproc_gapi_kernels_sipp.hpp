// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: drop SIPP from this file name

#pragma once

#if defined(__arm__) || defined(__aarch64__)
#include <opencv2/gapi.hpp>

namespace InferenceEngine {
namespace gapi {
namespace preproc {
namespace sipp {
cv::gapi::GKernelPackage kernels();
}  // namespace sipp

namespace m2i {
cv::gapi::GKernelPackage kernels();  // TODO: Remove (Stub)
}  // namespace m2i

}  // namespace preproc
}  // namespace gapi
}  // namespace InferenceEngine
#endif
