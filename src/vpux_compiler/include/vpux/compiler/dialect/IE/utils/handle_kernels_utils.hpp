//
// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/factors.hpp"

namespace vpux {
namespace IE {

bool hasSupportedKernels(ArrayRef<int64_t> kernelSize);
bool isPoolingKernelSizeValid(int64_t kernelSize);
Optional<Factors> getFactors(int64_t kernelSize, int64_t& padValue);
Optional<Factors> getFactorsWithSupportedLarger(int64_t kernelSize);

}  // namespace IE
}  // namespace vpux
