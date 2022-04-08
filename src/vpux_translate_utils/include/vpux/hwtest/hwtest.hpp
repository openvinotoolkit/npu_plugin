//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <mlir/Translation.h>

namespace vpux {

mlir::OwningModuleRef importHWTEST(llvm::StringRef sourceJson, mlir::MLIRContext* ctx);

}  // namespace vpux
