//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_CONST_PASSES
#define VPUX_COMPILER_DIALECT_CONST_PASSES

include "mlir/Pass/PassBase.td"

//
// ConstantFolding
//

def ConstantFolding : PassBase<"constant-folding", "vpux::FunctionPass"> {
    let summary = "Constant folding";

    let description = [{
        This pass performs constant folding.
    }];

    let constructor = "vpux::Const::createConstantFoldingPass()";

    let dependentDialects = [
        "vpux::Const::ConstDialect"
    ];
}

#endif
