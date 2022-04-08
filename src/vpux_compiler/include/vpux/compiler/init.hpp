//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/IR/Dialect.h>

namespace vpux {

void registerDialects(mlir::DialectRegistry& registry);

}  // namespace vpux
