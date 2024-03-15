//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/pattern_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IGreedilyPatternStrategy> createWrapVPUOpsInNCEClusterTilingStrategyGetter(
        mlir::func::FuncOp funcOp, bool enableExplicitDistributedTensorAttr);

}  // namespace vpux::VPU
