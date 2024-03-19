//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Dialect.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {

void registerDialects(mlir::DialectRegistry& registry);
void registerCommonInterfaces(mlir::DialectRegistry& registry, bool enableDummyOp = false);

}  // namespace vpux
