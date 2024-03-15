//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {

int64_t getPositiveAxisInd(mlir::IntegerAttr axisIndAttr, int64_t rank);

mlir::FailureOr<int64_t> getConstValue(mlir::Value input);
mlir::FailureOr<SmallVector<int64_t>> getConstArrValue(mlir::Value input);
mlir::FailureOr<int64_t> getConstOrAttrValue(mlir::Value input, mlir::IntegerAttr attr);
mlir::FailureOr<SmallVector<int64_t>> getConstOrArrAttrValue(mlir::Value input, mlir::ArrayAttr attr);

}  // namespace vpux
