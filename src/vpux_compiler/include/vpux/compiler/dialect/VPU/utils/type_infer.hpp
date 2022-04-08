//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes);
void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes);

}  // namespace VPU
}  // namespace vpux
