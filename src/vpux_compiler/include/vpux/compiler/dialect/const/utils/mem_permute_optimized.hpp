//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux {
namespace Const {
namespace details {

// checks capability to perform trivial NCHW<->NHWC and NDHWC<->NCDHW
bool isOptimizedTransformationSupported(vpux::Const::Content& input, vpux::NDTypeInterface outType);

//
// Performs specialized supported transformations
//
void memPermuteTransformationOptimized(vpux::Const::Content& input, vpux::Const::Content& output);

}  // namespace details
}  // namespace Const
}  // namespace vpux
