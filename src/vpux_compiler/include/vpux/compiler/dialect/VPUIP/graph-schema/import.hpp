//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace vpux {
namespace VPUIP {

mlir::OwningOpRef<mlir::ModuleOp> importBlob(mlir::MLIRContext* ctx, const std::vector<char>& blob,
                                             Logger log = Logger::global());

}  // namespace VPUIP
}  // namespace vpux
