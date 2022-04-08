//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/Pass/PassManager.h>

namespace vpux {

void addDotPrinter(mlir::PassManager& pm, mlir::StringRef options);

}  // namespace vpux
