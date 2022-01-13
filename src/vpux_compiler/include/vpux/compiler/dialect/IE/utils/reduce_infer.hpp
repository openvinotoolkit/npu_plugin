//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult inferReduceReturnTypeComponents(mlir::Location loc, mlir::Value input, bool keepDims,
                                                    SmallVector<int64_t>& axes,
                                                    SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes);
}  // namespace IE
}  // namespace vpux
