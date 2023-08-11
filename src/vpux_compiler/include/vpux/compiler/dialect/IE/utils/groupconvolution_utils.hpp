//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv);

}  // namespace IE
}  // namespace vpux
