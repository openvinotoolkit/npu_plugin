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

#ifndef VPUX_COMPILER_DIALECT_VPUIP
#define VPUX_COMPILER_DIALECT_VPUIP

include "mlir/IR/OpBase.td"

def VPUIP_Dialect : Dialect {
    let summary = "VPU NN RunTime Dialect";

    let description = [{
The **VPUIP Dialect** represents NN RunTime IR in terms of MLIR framework.

It allows to work with the graph schema inside MLIR framework:

* Validate it.
* Perform additional low level transformations/optimizations.

It handles such VPU-specifics as:

* Memory/executors hierarchy.
* HW barriers notion.
* Supported operation set.
    }];

    let name = "VPUIP";

    let cppNamespace = "vpux::VPUIP";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::IE::IEDialect",
        "vpux::IERT::IERTDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
