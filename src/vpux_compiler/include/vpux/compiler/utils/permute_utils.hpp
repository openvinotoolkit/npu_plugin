//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {

MemShape applyPerm(MemShapeRef memShape, mlir::AffineMap memPerm);

bool isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm);

mlir::AffineMap getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx);

}  // namespace vpux
