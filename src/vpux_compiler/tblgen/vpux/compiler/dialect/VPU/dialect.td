//
// Copyright Intel Corporation.
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

#ifndef VPUX_COMPILER_DIALECT_VPU
#define VPUX_COMPILER_DIALECT_VPU

include "mlir/IR/OpBase.td"

def VPU_Dialect : Dialect {
    let summary = "VPU high-level ISA Dialect";

    let description = [{
        The **VPU Dialect** represents high level ISA for VPU architectures.

        It is type compatible with **IE Dialect**, implements its interfaces and can be used along side with it.
        The operations in the **VPU Dialect** are pure functional and works on tensor level.
    }];

    let name = "VPU";

    let cppNamespace = "vpux::VPU";

    let extraClassDeclaration = [{
        void registerTypes();
    }];

    let dependentDialects = [
        "vpux::Const::ConstDialect",
        "vpux::IE::IEDialect",
        "mlir::StandardOpsDialect",
        "mlir::quant::QuantizationDialect"
    ];
}

#endif
