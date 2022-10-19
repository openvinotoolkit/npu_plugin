//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/stride_reqs.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/preprocessing.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

//
// DW Convolution utility
//

mlir::Value alignDepthWiseWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

//
// CM Convolution utility
//

mlir::Value alignChannelMajorWeightsTensor(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value origFilter);

//
// Reduce ops utility
//

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes);

//
// Permute ops utility
//

void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes);

}  // namespace VPU
}  // namespace vpux
