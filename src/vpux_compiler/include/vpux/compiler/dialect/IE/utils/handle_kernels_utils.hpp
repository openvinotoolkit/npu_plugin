//
// Copyright © 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

bool hasSupportedKernels(ArrayRef<int64_t> kernelSize);
bool isGlobalPoolingKernelSupported(mlir::Operation* op);

}  // namespace IE
}  // namespace vpux
