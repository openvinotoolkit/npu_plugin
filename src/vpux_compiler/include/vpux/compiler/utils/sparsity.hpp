//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"

namespace vpux {

// Get sparsify value and update element type to storage type
int64_t getSparsifyValue(mlir::Type& inputElementType);
SmallVector<int64_t> getNumActualElements(const Const::Content& content, mlir::Type elementType);

}  // namespace vpux
