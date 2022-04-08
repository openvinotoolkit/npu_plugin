//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPU37XX
#define VPUX_COMPILER_DIALECT_VPU37XX

include "mlir/IR/OpBase.td"

def VPU37XX_Dialect : Dialect {
    let summary = "VPU 37XX HW specifics";

    let description = [{
        The **37XX** encapsulates VPU2.7/37XX specific information.

    }];

    let name = "VPU37XX";

    let cppNamespace = "vpux::VPU37XX";

    let dependentDialects = [
        "vpux::VPUIPRegMapped::VPUIPRegMappedDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif
