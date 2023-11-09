//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPURT
#define VPUX_COMPILER_DIALECT_VPURT

include "mlir/IR/OpBase.td"

def VPURT_Dialect : Dialect {
    let summary = "VPU NN RunTime Scheduling Dialect";

    let description = [{
The **VPURT Dialect** represents NN execution RunTime IR in terms of MLIR framework.

It handles such VPU-specifics as:

* HW barriers notion.

[./VPURT/_ops_interfaces.md]
    }];

    let name = "VPURT";

    let cppNamespace = "vpux::VPURT";

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "vpux::IERT::IERTDialect",
        "vpux::VPU::VPUDialect",
        "vpux::VPUIP::VPUIPDialect",
    ];

    let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif
