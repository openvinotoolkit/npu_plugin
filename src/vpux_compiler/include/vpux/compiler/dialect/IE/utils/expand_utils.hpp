//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {
bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* op);
}  // namespace IE
}  // namespace vpux
