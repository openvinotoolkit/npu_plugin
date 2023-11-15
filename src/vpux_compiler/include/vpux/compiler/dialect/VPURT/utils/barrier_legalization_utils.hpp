//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/ops.hpp"

namespace vpux {
namespace VPURT {

void postProcessBarrierOps(mlir::func::FuncOp func);

}  // namespace VPURT
}  // namespace vpux
