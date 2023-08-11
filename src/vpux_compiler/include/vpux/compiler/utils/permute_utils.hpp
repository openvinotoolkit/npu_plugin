//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {

MemShape applyPerm(MemShapeRef memShape, mlir::AffineMap memPerm);

bool isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialReorder(IE::ReorderOp origOp);

mlir::AffineMap getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx);
DimsOrder applyPermutation(const DimsOrder lhs, const DimsOrder rhs);

}  // namespace vpux
