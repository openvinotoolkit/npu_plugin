//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/shave_kernel_info.hpp"

namespace vpux {
namespace VPU {

std::unique_ptr<VPU::ShaveKernelInfo> getShaveKernelInfo(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
