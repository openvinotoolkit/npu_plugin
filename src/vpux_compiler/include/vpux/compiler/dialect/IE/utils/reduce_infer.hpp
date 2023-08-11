//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/ops_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

namespace IE {

mlir::LogicalResult inferReduceReturnTypeComponents(mlir::Location loc, mlir::Value input, bool keepDims,
                                                    SmallVector<int64_t>& axes,
                                                    SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes);
void inferReduceLayoutInfo(mlir::Operation* op, IE::LayerLayoutInfo& info);
DimsOrder calculateReducedOutputLayout(DimsOrder inputDimOrder, mlir::SmallVector<int64_t> axes);
}  // namespace IE
}  // namespace vpux
