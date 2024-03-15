//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/Timing.h>

namespace vpux {
namespace ELFNPU37XX {

mlir::OwningOpRef<mlir::ModuleOp> importELF(mlir::MLIRContext* ctx, const std::string& elfFileName,
                                            Logger log = Logger::global());

}  // namespace ELFNPU37XX
}  // namespace vpux
