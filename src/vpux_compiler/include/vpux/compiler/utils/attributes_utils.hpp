//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {

int64_t getPositiveAxisInd(mlir::IntegerAttr axisIndAttr, int64_t rank);

}  // namespace vpux
