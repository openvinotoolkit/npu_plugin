//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

    let dependentDialects = [
        "mlir::StandardOpsDialect",
        "vpux::IERT::IERTDialect",
        "vpux::VPUIP::VPUIPDialect",
    ];
}

#endif
