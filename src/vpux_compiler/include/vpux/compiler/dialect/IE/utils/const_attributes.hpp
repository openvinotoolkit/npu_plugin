//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getIntArrayAttrValue(mlir::Value operand);
mlir::ArrayAttr getFloatArrayAttrValue(mlir::Value operand);
mlir::IntegerAttr getIntAttrValue(mlir::Value operand, mlir::PatternRewriter& rewriter);
mlir::FailureOr<Const::DeclareOp> getConstParentOp(mlir::Value input);

}  // namespace IE
}  // namespace vpux
