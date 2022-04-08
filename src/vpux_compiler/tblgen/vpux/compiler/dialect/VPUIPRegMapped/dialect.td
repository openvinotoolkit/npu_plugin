//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED

include "mlir/IR/OpBase.td"

def VPUIPRegMapped_Dialect : Dialect {
    let summary = "VPU NN Register Mapped RunTime Dialect";

    let description = [{
The **VPUIPRegMapped Dialect** represents NN RunTime IR together with RegMapped
    constructs in terms of the MLIR framework.

It allows to work with the graph schema inside the MLIR framework in order to:

* Validate it.
* Perform additional low level transformations/optimizations.

It handles such VPU-specifics as:

* Memory/executors hierarchy.
* HW barriers notion.
* Supported operation set.

Again, it represents also the register mapped configuration of the hardware registers.

[./VPUIPRegMapped/_ops_interfaces.md]
    }];

    let name = "VPUIPRegMapped";

    let cppNamespace = "vpux::VPUIPRegMapped";

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::ELF::ELFDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        void registerAttributes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
