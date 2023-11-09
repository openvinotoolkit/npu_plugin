//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/ops.hpp"

namespace vpux {
namespace IE {

void propagateElementTypeDown(LayerDataInfo<mlir::Type>& info);
void propagateElementTypeUp(LayerDataInfo<mlir::Type>& info);

}  // namespace IE
}  // namespace vpux
