//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::DeconvolutionOp deconv);
mlir::LogicalResult canConvertDeconvToConv(IE::DeconvolutionOp deconv);

}  // namespace IE
}  // namespace vpux
