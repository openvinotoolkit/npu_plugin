//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {
mlir::ArrayAttr getNewConcatOffsetsParameters(mlir::ArrayAttr oldOffsets, mlir::ArrayAttr dimsMappingAttr,
                                              mlir::OperandRange oldInputs, ArrayRef<vpux::ShapeRef> newInputShapes,
                                              ShapeRef reshapeShape, mlir::DenseSet<int64_t> modifiedAxes);
mlir::DenseSet<int64_t> getConcatModifiedAxis(IE::ConcatOp origOp);
SmallVector<int64_t> calculateInputShapeAfterSwitchConcatAndAffineReshape(mlir::Value input, IE::ConcatOp concatOp,
                                                                          IE::AffineReshapeOp reshapeOp);
}  // namespace IE
}  // namespace vpux
