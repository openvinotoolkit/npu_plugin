//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

//
// getFirstUser
//

mlir::Operation* getFirstUser(mlir::Value output);

//
// isBufAllocOp
//

bool isBufAllocOp(mlir::Operation* op);

//
// getModuleOp
//

mlir::ModuleOp getModuleOp(mlir::Operation* op);

}  // namespace vpux
