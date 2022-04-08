//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIP
#define VPUX_COMPILER_DIALECT_VPUIP

include "mlir/IR/OpBase.td"

def VPUIP_Dialect : Dialect {
    let summary = "VPU NN Execution RunTime Dialect";

    let description = [{
The **VPUIP Dialect** represents NN RunTime IR in terms of MLIR framework.

It allows to work with the graph schema inside MLIR framework:

* Validate it.
* Perform additional low level transformations/optimizations.

It handles such VPU-specifics as:

* Memory/executors hierarchy.
* Supported operation set.

[./VPUIP/_ops_interfaces.md]
    }];

    let name = "VPUIP";

    let cppNamespace = "vpux::VPUIP";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::IE::IEDialect",
        "vpux::VPU::VPUDialect",
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();

        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
