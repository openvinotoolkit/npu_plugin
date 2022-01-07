//
// Copyright (C) 2022 Intel Corporation.
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
        "vpux::IE::IEDialect",
        "vpux::IERT::IERTDialect",
        "vpux::ELF::ELFDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];

    let extraClassDeclaration = [{
        void registerTypes();
        static void setupExtraInterfaces(mlir::DialectRegistry& registry);
    }];
}

#endif
